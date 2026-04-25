#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA34 trend filter and volume spike confirmation.
Targets 7-25 trades/year by requiring: 1) price breaks weekly R1/S1 levels, 2) aligned with 1w EMA34 trend,
3) volume > 1.5x 20-day average volume. Uses 1d timeframe to minimize fee drag and capture significant weekly moves.
The volume spike filter ensures breakouts have conviction, while the 1w EMA34 trend filter avoids counter-trend trades.
Works in both bull and bear markets by only trading with the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Camarilla pivots and EMA34 (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1w levels to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # 1w EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 20-day average volume for volume spike filter
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1w previous data (1) + 1w EMA34 (34) + 20d avg volume (20)
    start_idx = 34 + 20 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(avg_vol_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation: volume > 1.5x 20-day average
        volume_spike = curr_volume > 1.5 * avg_vol_20[i]
        
        if position == 0:
            # Look for entry signals with trend alignment and volume confirmation
            # Long breakout: price breaks above R1 with uptrend and volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike
            # Short breakout: price breaks below S1 with downtrend and volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike
            
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
            # Long position: exit if price breaks below S1 (mean reversion) or trend changes to downtrend
            if curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above R1 (mean reversion) or trend changes to uptrend
            if curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0