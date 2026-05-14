# Strategy: 4h_WilliamsAlligator_12hEMA_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.248 | +33.9% | -10.3% | 257 | PASS |
| ETHUSDT | 0.205 | +32.1% | -21.2% | 252 | PASS |
| SOLUSDT | 1.208 | +255.1% | -23.6% | 238 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.385 | -9.4% | -12.6% | 91 | FAIL |
| ETHUSDT | 0.265 | +9.9% | -11.7% | 74 | PASS |
| SOLUSDT | -0.330 | -1.9% | -13.8% | 84 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h EMA trend filter and volume confirmation.
# Uses Williams Alligator (Jaw: 13-period smoothed median, Teeth: 8-period, Lips: 5-period)
# to identify trends. Long when Lips > Teeth > Jaw with volume and 12h EMA up.
# Short when Lips < Teeth < Jaw with volume and 12h EMA down.
# Designed to capture trends in both bull and bear markets by following 12h EMA.
# Williams Alligator uses smoothed medians (SMMA) which reduces whipsaw vs EMA.
name = "4h_WilliamsAlligator_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Williams Alligator: SMMA of median price
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - Williams Alligator uses this"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current) / Period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw: 13-period SMMA
    teeth = smma(median_price, 8)  # Teeth: 8-period SMMA
    lips = smma(median_price, 5)   # Lips: 5-period SMMA
    
    # 12h EMA trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume + 12h EMA up
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                vol_confirm[i] and price > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume + 12h EMA down
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  vol_confirm[i] and price < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth) or price below 12h EMA
            if lips[i] < teeth[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth) or price above 12h EMA
            if lips[i] > teeth[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 01:20
