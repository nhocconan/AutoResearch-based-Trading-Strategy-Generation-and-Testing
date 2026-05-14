# Strategy: 6h_WilliamsAlligator_DailyTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.064 | +16.5% | -12.0% | 123 | FAIL |
| ETHUSDT | 0.052 | +21.3% | -16.5% | 105 | PASS |
| SOLUSDT | 0.757 | +114.2% | -25.5% | 91 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.169 | +8.1% | -9.7% | 38 | PASS |
| SOLUSDT | -0.394 | -2.9% | -13.8% | 35 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day trend filter and volume confirmation
# Uses Alligator's Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
# Long when Lips > Teeth > Jaw (bullish alignment) + daily EMA(50) uptrend + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + daily EMA(50) downtrend + volume spike
# Alligator identifies trend absence/presence; alignment filters whipsaws in ranging markets
# Daily trend ensures higher timeframe momentum alignment
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with specific periods
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (blue line)
    teeth = smma(close, 8)  # Teeth (red line)
    lips = smma(close, 5)   # Lips (green line)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or np.isnan(jaw[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        lip = lips[i]
        tee = teeth[i]
        jaw_val = jaw[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + daily uptrend + volume spike
            if lip > tee and tee > jaw_val and close[i] > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + daily downtrend + volume spike
            elif lip < tee and tee < jaw_val and close[i] < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alignment breaks OR daily trend turns down
            if not (lip > tee and tee > jaw_val) or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alignment breaks OR daily trend turns up
            if not (lip < tee and tee < jaw_val) or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-08 13:11
