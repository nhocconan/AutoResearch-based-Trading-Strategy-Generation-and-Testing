# Strategy: 4h_Camarilla_R3S3_Reversal_1dEMA34_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.335 | +40.3% | -11.2% | 297 | PASS |
| ETHUSDT | 0.144 | +27.2% | -14.5% | 327 | PASS |
| SOLUSDT | 0.376 | +56.8% | -32.1% | 356 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.559 | -1.6% | -10.1% | 129 | FAIL |
| ETHUSDT | 1.218 | +33.2% | -10.5% | 114 | PASS |
| SOLUSDT | 1.021 | +30.1% | -10.4% | 115 | PASS |

## Code
```python
#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla Pivot Reversal with 1-day EMA trend filter and volume confirmation.
Trades reversals at Camarilla R3/S3 levels (strong support/resistance) in the direction of the daily EMA trend.
Uses volume spike to confirm institutional interest at key levels. Designed for low trade frequency
(20-50 trades/year) to minimize fee drag and work in both bull and bear markets by aligning with
higher timeframe trend and using mean-reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = close + range_val * 1.1 / 2
    s3 = close - range_val * 1.1 / 2
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_r3, camarilla_s3 = calculate_camarilla(high_1d, low_1d, close_1d_arr)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price crosses above S3 support with uptrend bias
            if close[i] > camarilla_s3_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R3 resistance with downtrend bias
            elif close[i] < camarilla_r3_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below S3 or closes below daily EMA
                if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above R3 or closes above daily EMA
                if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Reversal_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 18:40
