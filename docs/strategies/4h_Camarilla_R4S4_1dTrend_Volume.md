# Strategy: 4h_Camarilla_R4S4_1dTrend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.116 | +13.7% | -15.2% | 161 | FAIL |
| ETHUSDT | 0.123 | +25.9% | -13.0% | 152 | PASS |
| SOLUSDT | 0.831 | +123.2% | -17.8% | 121 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.174 | +27.6% | -8.2% | 50 | PASS |
| SOLUSDT | 0.076 | +6.4% | -11.4% | 44 | PASS |

## Code
```python
# 4h_Camarilla_R4S4_1dTrend_Volume
# Strategy combines 1-day Camarilla R4/S4 levels with 1-day trend filter and volume confirmation.
# Enters long when price breaks above R4 with daily uptrend and volume spike, short when price breaks below S4 with daily downtrend and volume spike.
# Exits on trend reversal or price crossing opposite level (R4<->S4). Uses daily timeframe for both levels and trend to avoid look-ahead.
# Designed to work in both bull and bear markets by aligning with daily trend. Target: 20-50 trades/year to minimize fee drag.
# This version adds volume confirmation and uses discrete position sizing to reduce churn.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R4S4_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla R4, S4 levels: (H-L)*1.1/2
    camarilla_range = (high_1d - low_1d) * 1.1 / 2
    r4_level = close_1d_vals + camarilla_range
    s4_level = close_1d_vals - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_level)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_level)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for EMA20 (1d) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1d_val = ema20_1d_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above R4 + 1d uptrend + volume spike
            if close[i] > r4 and close[i] > ema20_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below S4 + 1d downtrend + volume spike
            elif close[i] < s4 and close[i] < ema20_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S4 or 1d trend turns down
            if close[i] < s4 or close[i] < ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above R4 or 1d trend turns up
            if close[i] > r4 or close[i] > ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-09 04:28
