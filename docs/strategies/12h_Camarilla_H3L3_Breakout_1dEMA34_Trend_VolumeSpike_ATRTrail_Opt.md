# Strategy: 12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrail_Opt

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.124 | +25.3% | -5.3% | 74 | PASS |
| ETHUSDT | 0.174 | +27.9% | -7.0% | 64 | PASS |
| SOLUSDT | 0.176 | +29.8% | -19.7% | 66 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.035 | -2.3% | -7.7% | 33 | FAIL |
| ETHUSDT | 0.784 | +15.9% | -4.6% | 24 | PASS |
| SOLUSDT | -0.247 | +2.4% | -12.1% | 22 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike + ATR Trailing Stop (Optimized)
Hypothesis: Camarilla H3L3 levels on 12h act as strong support/resistance. Breakouts above H3 or below L3 with volume confirmation (>2x 20-period volume MA) capture momentum. 1d EMA34 filter ensures alignment with daily trend. ATR trailing stop (3x ATR) manages risk and reduces whipsaw. Optimized for higher trade frequency and better risk-adjusted returns across BTC, ETH, and SOL in both bull and bear markets.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 days for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for trailing stop (12h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Camarilla levels for 12h (using previous 12h bar's high, low, close)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous bar's HLC to calculate today's levels (no look-ahead)
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        rang = phigh - plow
        camarilla_h3[i] = pclose + rang * 1.1 / 4
        camarilla_l3[i] = pclose - rang * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need enough for EMA34, volume MA, ATR, and Camarilla
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        camarilla_h3_val = camarilla_h3[i]
        camarilla_l3_val = camarilla_l3[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla H3/L3 levels
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_h3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_l3_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: 3 * ATR below highest since entry
            trailing_stop = highest_since_entry - 3.0 * atr_val
            # Exit conditions: price closes below Camarilla L3 OR trailing stop hit OR EMA34 trend turns down
            if curr_close < camarilla_l3_val or curr_close < trailing_stop or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: 3 * ATR above lowest since entry
            trailing_stop = lowest_since_entry + 3.0 * atr_val
            # Exit conditions: price closes above Camarilla H3 OR trailing stop hit OR EMA34 trend turns up
            if curr_close > camarilla_h3_val or curr_close > trailing_stop or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrail_Opt"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-25 04:04
