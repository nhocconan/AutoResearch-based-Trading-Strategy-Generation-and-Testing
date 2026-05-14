# Strategy: 12h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.085 | +17.9% | -8.9% | 71 | FAIL |
| ETHUSDT | 0.038 | +21.8% | -7.3% | 59 | PASS |
| SOLUSDT | 0.325 | +43.0% | -23.9% | 61 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.137 | +7.4% | -5.6% | 24 | PASS |
| SOLUSDT | -1.256 | -10.1% | -17.3% | 23 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Camarilla H3/L3 breakouts on 12h with 1d EMA34 trend filter and volume confirmation.
H3/L3 levels provide stronger support/resistance than R1/S1, reducing false breakouts.
1d EMA trend filter ensures alignment with long-term momentum.
Volume spike confirms institutional participation.
Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag in BTC/ETH ranging/bear markets.
Works in both bull (breakouts with trend) and bear (fade breakdowns in downtrend) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price_1d = (h_1d + l_1d + c_1d) / 3.0
    range_1d = h_1d - l_1d
    camarilla_h3_1d = typical_price_1d + (range_1d * 1.1 / 4.0)
    camarilla_l3_1d = typical_price_1d - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Volume confirmation: 12h volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla H3 + 1d uptrend + volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: price breaks below Camarilla L3 + 1d downtrend + volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla L3 (stop) OR 1d trend turns bearish
            if (close[i] <= camarilla_l3_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H3 (stop) OR 1d trend turns bullish
            if (close[i] >= camarilla_h3_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-25 15:40
