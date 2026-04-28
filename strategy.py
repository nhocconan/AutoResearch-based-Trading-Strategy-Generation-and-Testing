#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot (R1/S1) breakout on 12h with 1d EMA34 trend filter and volume spike confirmation.
Targets 15-35 trades/year to minimize fee drift. Works in bull via breakouts with trend, in bear via fade
of false breaks when price rejects R1/S1 and reverts to mean within range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend and pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for current day
        day_idx = i // 2  # 2 = 12h bars per day
        if day_idx < 1:
            signals[i] = 0.0
            continue
            
        prev_day_idx = day_idx - 1
        if prev_day_idx >= len(df_1d):
            signals[i] = 0.0
            continue
            
        ph = df_1d['high'].iloc[prev_day_idx]
        pl = df_1d['low'].iloc[prev_day_idx]
        pc = df_1d['close'].iloc[prev_day_idx]
        
        range_val = ph - pl
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        r1 = pc + (range_val * 1.1 / 12)
        s1 = pc - (range_val * 1.1 / 12)
        
        # Trend and volume
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_confirm = volume[i] > (2.0 * vol_ma_20)
        
        # Breakout and rejection
        long_breakout = close[i] > r1
        short_breakout = close[i] < s1
        long_reject = close[i] < r1 and close[i] > (r1 + s1)/2  # Reject above midpoint
        short_reject = close[i] > s1 and close[i] < (r1 + s1)/2  # Reject below midpoint
        
        # Regime: avoid strong trends (use price vs EMA distance)
        price_vs_ema = abs(close[i] - ema_34_1d_aligned[i]) / ema_34_1d_aligned[i]
        strong_trend = price_vs_ema > 0.08  # >8% deviation
        
        # Entry: fade rejection in ranging, break with trend in trending
        if not strong_trend:  # ranging market
            long_entry = vol_confirm and long_reject and (position <= 0)
            short_entry = vol_confirm and short_reject and (position >= 0)
        else:  # trending market
            long_entry = vol_confirm and trend_up and long_breakout and (position <= 0)
            short_entry = vol_confirm and trend_down and short_breakout and (position >= 0)
        
        # Exit: opposite signal or loss of momentum
        long_exit = (short_breakout and trend_down) or (not trend_up and position == 1)
        short_exit = (long_breakout and trend_up) or (not trend_down and position == -1)
        
        if long_entry:
            signals[i] = 0.25
            position = 1
        elif short_entry:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0