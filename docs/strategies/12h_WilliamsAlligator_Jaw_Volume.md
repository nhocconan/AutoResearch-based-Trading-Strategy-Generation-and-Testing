# Strategy: 12h_WilliamsAlligator_Jaw_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.186 | +5.9% | -17.6% | 95 | FAIL |
| ETHUSDT | 0.019 | +16.6% | -18.2% | 94 | PASS |
| SOLUSDT | 1.192 | +295.7% | -26.5% | 94 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.589 | +18.3% | -9.0% | 29 | PASS |
| SOLUSDT | 0.133 | +7.1% | -11.5% | 29 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams Alligator with volume confirmation.
# Long when price > Alligator's Jaw (13-period SMMA) with volume > 1.3x average.
# Short when price < Alligator's Jaw with volume > 1.3x average.
# Exit when price crosses the Jaw in opposite direction.
# Uses Alligator for trend (SMMA = Smoothed Moving Average), volume for confirmation.
# Target: 12-37 trades/year to avoid fee drag. Works in bull/bear via trend-following.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (blue): 13-period SMMA, 8 bars ahead
    # Teeth (red): 8-period SMMA, 5 bars ahead
    # Lips (green): 5-period SMMA, 3 bars ahead
    # We use Jaw (13,8) as the main trend indicator
    
    def smoothed_mma(data, period):
        """Smoothed Moving Average (SMMA)"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_1d = smoothed_mma(close_1d, 13)
    # We don't need teeth/lips for this strategy, using only Jaw
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 13-period Jaw and 20-period volume MA
    start_idx = max(13, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.3 * vol_avg
        
        if position == 0:
            # Long: price > Jaw with volume confirmation
            if price > jaw_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price < Jaw with volume confirmation
            elif price < jaw_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Jaw
            if price < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Jaw
            if price > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Jaw_Volume"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:46
