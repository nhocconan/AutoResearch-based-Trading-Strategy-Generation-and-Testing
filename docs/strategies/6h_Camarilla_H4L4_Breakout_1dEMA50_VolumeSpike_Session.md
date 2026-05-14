# Strategy: 6h_Camarilla_H4L4_Breakout_1dEMA50_VolumeSpike_Session

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.176 | +17.8% | -4.6% | 162 | FAIL |
| ETHUSDT | 0.107 | +24.2% | -5.8% | 157 | PASS |
| SOLUSDT | 0.254 | +31.5% | -12.1% | 126 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.784 | +12.2% | -4.7% | 59 | PASS |
| SOLUSDT | 0.960 | +13.0% | -2.3% | 46 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe strategy using 12h Camarilla pivot levels with 1d volume spike and EMA50 trend filter.
- Uses 12h for signal direction (Camarilla H4/L4 breakout) and 1d for trend filter (EMA50) and volume confirmation (>2.0x average)
- 6h only for entry timing precision to reduce whipsaw
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 2.0x 24-period average (strict for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 12h Camarilla pivot levels (H4, L4, H3, L3)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    # Calculate range
    range_12h = high_12h - low_12h
    # Camarilla levels
    h4_12h = pp_12h + range_12h * 1.1 / 2
    l4_12h = pp_12h - range_12h * 1.1 / 2
    h3_12h = pp_12h + range_12h * 1.1 / 4
    l3_12h = pp_12h - range_12h * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (use prior completed 12h bar)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(h4_12h_aligned[i]) or
            np.isnan(l4_12h_aligned[i]) or
            np.isnan(h3_12h_aligned[i]) or
            np.isnan(l3_12h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up_h4 = close[i] > h4_12h_aligned[i-1]  # Close above prior 12h H4
        breakout_down_l4 = close[i] < l4_12h_aligned[i-1]  # Close below prior 12h L4
        
        if position == 0:
            # Long: 12h Camarilla H4 breakout up AND price > 1d EMA50 AND volume confirmation AND in session
            if breakout_up_h4 and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h Camarilla L4 breakout down AND price < 1d EMA50 AND volume confirmation AND in session
            elif breakout_down_l4 and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 12h Camarilla L3 break down OR price < 1d EMA50 (trend flip)
            if close[i] < l3_12h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 12h Camarilla H3 break up OR price > 1d EMA50 (trend flip)
            if close[i] > h3_12h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_1dEMA50_VolumeSpike_Session"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-23 22:46
