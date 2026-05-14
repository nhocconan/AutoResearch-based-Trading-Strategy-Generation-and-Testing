# Strategy: 4h_Williams_Alligator_1dEMA34_Trend_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.255 | +33.9% | -11.1% | 218 | PASS |
| ETHUSDT | 0.107 | +24.6% | -14.5% | 220 | PASS |
| SOLUSDT | 1.078 | +199.1% | -23.7% | 202 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.992 | -5.0% | -8.9% | 80 | FAIL |
| ETHUSDT | 0.214 | +8.9% | -12.0% | 69 | PASS |
| SOLUSDT | -0.263 | -0.3% | -13.3% | 76 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume confirmation
    # Williams Alligator identifies trend direction and strength via SMMA crossovers
    # When Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
    # Entries require Alligator alignment + 1d EMA34 trend filter + volume spike
    # This combination filters weak signals and improves win rate in both bull/bear markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator components (using SMMA - smoothed moving average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: (prev*(period-1) + current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (13-period SMMA)
    teeth = smma(close, 8)  # Teeth (8-period SMMA)
    lips = smma(close, 5)   # Lips (5-period SMMA)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume spike + price above 1d EMA34
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume spike + price below 1d EMA34
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (trend weakening) or trend reversal vs 1d EMA34
            if position == 1:
                if lips[i] < teeth[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 07:23
