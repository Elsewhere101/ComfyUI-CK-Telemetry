import { app } from "../../../scripts/app.js";

const NODE_TYPES = new Set([
    "CK-Telemetry",
    "CKDispatchReport",
    "ComfyKitchenTelemetryImage",
    "ComfyKitchenTelemetryVideo",
]);
const REPORT_WIDGET = "report_text";
const SHOW_WIDGET = "show_node_report";
const REPORT_PROPERTY = "comfy_kitchen_dispatch_report";

function findNodeById(id) {
    const numericId = Number(id);
    return app.graph?.getNodeById?.(numericId) ||
        app.graph?._nodes_by_id?.[id] ||
        app.graph?._nodes_by_id?.[numericId];
}

function setReportWidgetVisibility(node) {
    const report = node.widgets?.find((w) => w.name === REPORT_WIDGET);
    const show = node.widgets?.find((w) => w.name === SHOW_WIDGET);
    if (!report || !show) return;

    const visible = Boolean(show.value);
    report.hidden = !visible;
    report.options = report.options || {};
    report.options.hidden = !visible;
    report.computeSize = visible ? undefined : () => [0, -4];
    if (report.element?.style) report.element.style.display = visible ? "" : "none";
    if (report.inputEl?.style) report.inputEl.style.display = visible ? "" : "none";
    node.setDirtyCanvas?.(true, true);
}

function setReportReadOnly(node) {
    const widget = node.widgets?.find((w) => w.name === REPORT_WIDGET);
    if (!widget) return;
    const root = widget.inputEl || widget.element;
    if (!root) return;
    const elements = root.matches?.("textarea, input")
        ? [root]
        : Array.from(root.querySelectorAll?.("textarea, input") || []);
    for (const element of elements) {
        element.readOnly = true;
        element.setAttribute("readonly", "readonly");
        element.title = "Generated diagnostic report; refreshed after execution.";
    }
}

function setReport(node, reportText) {
    if (typeof reportText !== "string") return;
    const widget = node.widgets?.find((w) => w.name === REPORT_WIDGET);
    if (!widget) return;

    widget.value = reportText;
    const input = widget.inputEl;
    if (input && "value" in input && input.value !== reportText) {
        input.value = reportText;
    }

    node.properties = node.properties || {};
    node.properties[REPORT_PROPERTY] = reportText;
    setReportWidgetVisibility(node);
    setReportReadOnly(node);
    node.setDirtyCanvas?.(true, true);
}

function restorePersistedReport(node) {
    const report = node.properties?.[REPORT_PROPERTY];
    if (typeof report === "string" && report.length) setReport(node, report);
}

function handleExecuted(detail) {
    const nodeId = detail?.display_node ?? detail?.node;
    const node = findNodeById(nodeId);
    if (!node || !NODE_TYPES.has(node.constructor?.comfyClass)) return;

    const text = detail?.output?.text?.[0];
    if (typeof text === "string") setReport(node, text);
}

function styleReportWidget(node) {
    const widget = node.widgets?.find((w) => w.name === REPORT_WIDGET);
    if (!widget) return;

    const root = widget.inputEl || widget.element;
    if (!root) return;
    const textarea = root.matches?.("textarea")
        ? root
        : root.querySelector?.("textarea");
    if (!textarea) return;

    const container = textarea.parentElement;
    if (container) {
        container.style.position = "relative";
        container.style.backgroundImage =
            'url("/extensions/ComfyUI-CK-Telemetry/bg.png")';
        container.style.backgroundRepeat = "repeat";
        container.style.backgroundSize = "362px 737px";
    }

    textarea.style.backgroundColor = "transparent";
    textarea.style.color = "#e4e4e7";
    textarea.style.fontFamily = "Consolas, 'Courier New', monospace";
    textarea.style.fontSize = "12px";
    textarea.style.lineHeight = "1.5";
    textarea.style.border = "1px solid #ff5722";
    textarea.style.borderRadius = "4px";
    textarea.style.padding = "8px";
    textarea.style.boxShadow = "inset 0 0 8px rgba(0,0,0,0.6)";
}

app.registerExtension({
    name: "ComfyKitchen.DispatchTelemetry",

    async nodeCreated(node) {
        if (!NODE_TYPES.has(node.constructor?.comfyClass)) return;
        const show = node.widgets?.find((w) => w.name === SHOW_WIDGET);
        if (show) {
            const original = show.callback;
            show.callback = function (value) {
                original?.apply(this, arguments);
                setReportWidgetVisibility(node);
                setReportReadOnly(node);
                styleReportWidget(node);
                requestAnimationFrame(() => styleReportWidget(node));
            };
        }
        restorePersistedReport(node);
        setReportWidgetVisibility(node);
        setReportReadOnly(node);
        styleReportWidget(node);
        requestAnimationFrame(() => styleReportWidget(node));
    },

    setup() {
        app.api.addEventListener("executed", ({ detail }) => handleExecuted(detail));
    },

    async afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) {
            if (NODE_TYPES.has(node.constructor?.comfyClass)) {
                restorePersistedReport(node);
                setReportWidgetVisibility(node);
                setReportReadOnly(node);
                styleReportWidget(node);
            }
        }
    },
});
