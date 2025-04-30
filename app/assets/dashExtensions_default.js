window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            const {
                style,
                colorProp
            } = context.hideout; // get props from hideout
            const value = feature.properties[colorProp]; // get value that determines the color
            const color_codes = {
                'South Fork Area': '#ff0000', // Red
                'Castaic Valley Area': '#00ff00', // Green
                'San Francisquto Canyon Area': '#0000ff', // Blue
                'Below Valencia WRP Area': '#ffff00', // Yellow
                'Mint Canyon Area': '#ff00ff', // Magenta
                'Bouquet Canyon Area': '#00ffff', // Cyan
                'Above Saugus WRP Area': '#ffa500', // Orange
                'Below Saugus WRP Area': '#800080' // Purple
            };
            style.fillColor = color_codes[value]; // set the fill color according to the class
            return style;
        },
        function1: function(feature, latlng) {
            return L.circleMarker(latlng, {
                fillOpacity: 0.8,
                radius: 5
            }); // render a circle marker
        },
        function2: function(feature, latlng) {
            const icon = new scatterIcon({
                opacity: 0,
                fillOpacity: 0,
            });
            return L.circleMarker(latlng, {
                icon: icon
            }); // render an invisibile cluster marker
        }
    }
});