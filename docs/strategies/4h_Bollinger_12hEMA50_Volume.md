# Strategy: 4h_Bollinger_12hEMA50_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.439 | +32.8% | -7.1% | 289 | PASS |
| ETHUSDT | 0.123 | +24.8% | -5.7% | 273 | PASS |
| SOLUSDT | 0.035 | +21.1% | -21.5% | 279 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.863 | +1.8% | -3.3% | 102 | FAIL |
| ETHUSDT | 1.178 | +17.7% | -3.2% | 86 | PASS |
| SOLUSDT | 0.049 | +6.4% | -5.8% | 90 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band breakout with 12-hour EMA trend filter and volume confirmation
# Long when price closes above upper Bollinger Band AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price closes below lower Bollinger Band AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Bollinger Bands (opposite band)
# Uses Bollinger Bands to capture volatility expansions, EMA for trend alignment, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Bollinger Bands on 4h (20-period, 2 std dev)
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Bollinger + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: close above upper BB + above 12h EMA50 + volume confirmation
            if (price > upper_band[i] and price > ema50_12h_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: close below lower BB + below 12h EMA50 + volume confirmation
            elif (price < lower_band[i] and price < ema50_12h_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes back inside Bollinger Bands (below upper band)
            if price < upper_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes back inside Bollinger Bands (above lower band)
            if price > lower_band[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-14 03:55
