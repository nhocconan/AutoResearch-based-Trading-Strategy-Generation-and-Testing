# Strategy: 6h_Camarilla_R4S4_Breakout_1dEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.371 | +40.7% | -14.4% | 61 | PASS |
| ETHUSDT | 0.084 | +22.9% | -15.8% | 58 | PASS |
| SOLUSDT | 0.152 | +26.6% | -37.4% | 51 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.673 | -1.7% | -8.0% | 20 | FAIL |
| ETHUSDT | 0.165 | +8.0% | -11.4% | 23 | PASS |
| SOLUSDT | -0.304 | -0.7% | -13.6% | 18 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1d trend filter and volume spike confirmation.
# Uses 1d Camarilla pivot levels for breakout detection (R4/S4 = strong breakout levels),
# filtered by 1d EMA50 trend and volume > 2.0x 20-period median.
# R4/S4 breaks indicate institutional interest and work in both bull (buy R4 breakouts with uptrend)
# and bear (sell S4 breakdowns with downtrend). Discrete position sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Camarilla_R4S4_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla pivot levels (R4, S4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    rang = high_1d - low_1d
    r4 = pivot + rang * 1.1
    s4 = pivot - rang * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50 and volume median
    start_idx = max(50, 20) + 1  # 51
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Camarilla breakout conditions (R4/S4 = strong breakout levels)
        breakout_up = curr_close > r4_aligned[i]   # break above R4
        breakout_down = curr_close < s4_aligned[i]  # break below S4
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R4 AND uptrend AND volume confirmation
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S4 AND downtrend AND volume confirmation
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below S4 (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above R4 (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-01 15:08
