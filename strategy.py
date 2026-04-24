#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakout for direction and 1h for entry timing.
- Uses 4h Camarilla H3/L3 levels as breakout thresholds (proven structure from DB toppers)
- 1d EMA34 as HTF trend filter to avoid counter-trend trades in bear markets
- Volume confirmation (>2.0x 20-bar average) to ensure institutional participation
- Session filter (08-20 UTC) to reduce noise trades outside active hours
- Fixed position size 0.20 to balance profit and drawdown control
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
- Works in bull/bear markets via 1d trend filter and high-probability breakout logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior 4h bar
    # Need to shift 4h data by 1 bar to avoid look-ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    h3_4h = pivot_4h + (range_4h * 1.1 / 4)  # H3 level (using 1.1 for slightly tighter breakout)
    l3_4h = pivot_4h - (range_4h * 1.1 / 4)  # L3 level
    
    # Align Camarilla levels to 1h timeframe
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms AND in session
            if volume_confirm:
                # Long breakout: price above H3 AND above 1d EMA34 (uptrend)
                if close[i] > h3_4h_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short breakout: price below L3 AND below 1d EMA34 (downtrend)
                elif close[i] < l3_4h_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR crosses below 1d EMA34
            if close[i] < l3_4h_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR crosses above 1d EMA34
            if close[i] > h3_4h_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0