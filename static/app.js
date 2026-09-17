let datasetUploaded = false;


// ==========================================================
// LOAD OVERVIEW
// ==========================================================

async function loadOverview() {

    const response =
        await fetch("/api/overview");

    const data =
        await response.json();

    if (!data.uploaded) {

        document
            .querySelectorAll("[data-value]")
            .forEach(element => {

                element.textContent = "—";

            });

        return;
    }

    datasetUploaded = true;

    setValue("rows", data.rows);
    setValue("columns", data.columns);
    setValue("missing", data.missing);
    setValue("duplicates", data.duplicates);

    setValue("total", data.total);
    setValue("dropouts", data.dropouts);
    setValue("retained", data.retained);

    if (data.risk_rate !== null) {

        setValue(
            "risk_rate",
            data.risk_rate + "%"
        );

    }
}


// ==========================================================
// SET VALUE
// ==========================================================

function setValue(name, value) {

    const element =
        document.querySelector(
            `[data-value="${name}"]`
        );

    if (element) {

        element.textContent =
            value ?? "—";

    }
}


// ==========================================================
// LOAD EDA
// ==========================================================

async function loadEDA() {

    const response =
        await fetch("/api/eda");

    const data =
        await response.json();

    if (!data.uploaded) {
        return;
    }

    renderModuleChart(data.module);
    renderAgeChart(data.age);
    renderEducationChart(data.education);
    renderResultChart(data.result);
    renderTargetChart(data.target);

    renderMissingTable(
        data.missing
    );

    renderOutlierTable(
        data.outliers
    );
}


// ==========================================================
// CHART CONFIG
// ==========================================================

function chartOptions() {

    return {

        responsive: true,

        maintainAspectRatio: false,

        plugins: {

            legend: {
                display: false
            }

        },

        scales: {

            x: {
                grid: {
                    display: false
                }
            },

            y: {
                beginAtZero: true,

                grid: {
                    color: "#edf0f6"
                }
            }

        }
    };
}


// ==========================================================
// MODULE CHART
// ==========================================================

function renderModuleChart(data) {

    const canvas =
        document.getElementById(
            "moduleChart"
        );

    if (!canvas) return;

    new Chart(
        canvas,
        {
            type: "bar",

            data: {

                labels: data.labels,

                datasets: [{
                    data: data.values,
                    borderRadius: 10,
                    borderWidth: 0
                }]
            },

            options: chartOptions()
        }
    );
}


// ==========================================================
// AGE CHART
// ==========================================================

function renderAgeChart(data) {

    const canvas =
        document.getElementById(
            "ageChart"
        );

    if (!canvas) return;

    new Chart(
        canvas,
        {

            type: "line",

            data: {

                labels: data.labels,

                datasets: [{
                    data: data.values,
                    tension: 0.4,
                    fill: true
                }]
            },

            options: chartOptions()
        }
    );
}


// ==========================================================
// EDUCATION CHART
// ==========================================================

function renderEducationChart(data) {

    const canvas =
        document.getElementById(
            "educationChart"
        );

    if (!canvas) return;

    new Chart(
        canvas,
        {

            type: "bar",

            data: {

                labels: data.labels,

                datasets: [{
                    data: data.values,
                    borderRadius: 8
                }]
            },

            options: {

                ...chartOptions(),

                indexAxis: "y"

            }
        }
    );
}


// ==========================================================
// RESULT CHART
// ==========================================================

function renderResultChart(data) {

    const canvas =
        document.getElementById(
            "resultChart"
        );

    if (!canvas) return;

    new Chart(
        canvas,
        {

            type: "doughnut",

            data: {

                labels: data.labels,

                datasets: [{
                    data: data.values
                }]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        display: true,
                        position: "bottom"
                    }

                }

            }
        }
    );
}


// ==========================================================
// TARGET
// ==========================================================

function renderTargetChart(data) {

    const canvas =
        document.getElementById(
            "targetChart"
        );

    if (!canvas ||
        !data.available) return;

    new Chart(
        canvas,
        {

            type: "doughnut",

            data: {

                labels: data.labels,

                datasets: [{
                    data: data.values
                }]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false

            }

        }
    );
}


// ==========================================================
// MISSING TABLE
// ==========================================================

function renderMissingTable(data) {

    const table =
        document.getElementById(
            "missingTable"
        );

    if (!table) return;

    if (!data.length) {

        table.innerHTML = `
            <tr>
                <td colspan="3">
                    No missing values 🎉
                </td>
            </tr>
        `;

        return;
    }

    table.innerHTML =
        data.map(row => `
            <tr>
                <td>${row.Feature}</td>
                <td>${row["Missing Values"]}</td>
                <td>${row.Percentage}%</td>
            </tr>
        `).join("");
}


// ==========================================================
// OUTLIER TABLE
// ==========================================================

function renderOutlierTable(data) {

    const table =
        document.getElementById(
            "outlierTable"
        );

    if (!table) return;

    table.innerHTML =
        data.map(row => `
            <tr>
                <td>${row.feature}</td>
                <td>${row.count}</td>
                <td>${row.percentage}%</td>
            </tr>
        `).join("");
}


// ==========================================================
// SEARCH
// ==========================================================

async function searchStudents(query) {

    const response =
        await fetch(
            "/api/students?q=" +
            encodeURIComponent(query)
        );

    const data =
        await response.json();

    const tbody =
        document.getElementById(
            "studentRows"
        );

    if (!tbody) return;

    tbody.innerHTML =
        data.map(row => {

            return `
                <tr>
                    <td>
                        ${row.id_student ?? "—"}
                    </td>

                    <td>
                        ${row.code_module ?? "—"}
                    </td>

                    <td>
                        ${row.gender ?? "—"}
                    </td>

                    <td>
                        ${row.age_band ?? "—"}
                    </td>

                    <td>
                        ${row.highest_education ?? "—"}
                    </td>

                    <td>
                        ${row.avg_assessment_score ?? "—"}
                    </td>

                    <td>
                        ${row.total_vle_clicks ?? "—"}
                    </td>

                    <td>
                        ${row.final_result ?? "—"}
                    </td>

                    <td>
                        ${row.dropout == 1
                            ? "Higher"
                            : "Lower"}
                    </td>
                </tr>
            `;

        }).join("");
}


// ==========================================================
// START
// ==========================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadOverview();
        loadEDA();

        const search =
            document.getElementById(
                "studentSearch"
            );

        if (search) {

            search.addEventListener(
                "input",
                event => {

                    searchStudents(
                        event.target.value
                    );

                }
            );

        }

    }
);