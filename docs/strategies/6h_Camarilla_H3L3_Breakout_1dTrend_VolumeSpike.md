# Strategy: 6h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.114 | +25.0% | -6.2% | 647 | PASS |
| ETHUSDT | 0.089 | +24.1% | -7.3% | 569 | PASS |
| SOLUSDT | 0.454 | +53.0% | -20.2% | 468 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.862 | -5.7% | -5.7% | 185 | FAIL |
| ETHUSDT | 0.498 | +11.5% | -7.7% | 191 | PASS |
| SOLUSDT | -0.280 | +2.6% | -6.5% | 177 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: 6-hour Camarilla H3/L3 breakout with 1-day EMA34 trend filter and volume spike confirmation.
Targets 12-37 trades/year by requiring: 1) price breaks daily H3/L3 levels (strong intraday breakout),
2) aligned with 1d EMA34 trend, 3) volume > 2.0x 20-period average. Uses 6h timeframe to minimize
fee drag while capturing significant moves in both bull and bear markets. H3/L3 levels provide
better signal quality than R1/S1 for lower timeframe strategies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla H3 and L3 levels (H3 = C + 1.1*(HL/2), L3 = C - 1.1*(HL/2))
    H3 = prev_close + 1.1 * prev_range * (1.0/2.0)
    L3 = prev_close - 1.1 * prev_range * (1.0/2.0)
    
    # Align 1d levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and previous day data (1)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment
            # Long breakout: price breaks above H3 with uptrend and volume confirmation
            long_breakout = (curr_close > H3_aligned[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below L3 with downtrend and volume confirmation
            short_breakout = (curr_close < L3_aligned[i]) and downtrend and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below L3 (mean reversion) or trend changes
            if curr_close < L3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above H3 (mean reversion) or trend changes
            if curr_close > H3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 10:41
