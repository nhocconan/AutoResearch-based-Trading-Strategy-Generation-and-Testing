#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Trade daily Camarilla H3/L3 breakouts only when 1-week EMA34 trend aligns, volume spikes (>2.0x 20-bar MA), and market is not choppy (Chop < 61.8 on 1d). 
H3/L3 levels offer stronger breakouts than R1/S1, reducing false signals. Works in bull markets (breakouts with trend) 
and bear markets (fades from extremes with volume). Chop filter avoids whipsaws in ranging markets. Discrete sizing 0.25 limits fee drag. 
Target 15-25 trades/year on 1d timeframe.
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3 from 1d
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 1d timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (14-period) on 1d - avoid choppy markets
    # Chop > 61.8 = ranging/choppy (avoid), Chop < 38.2 = trending (favor)
    tr_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_sum = pd.Series(tr_range).rolling(window=14, min_periods=14).sum().values
    true_range_sum = tr_sum
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(true_range_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[~np.isfinite(chop)] = 50  # default to neutral when undefined
    not_choppy = chop < 61.8  # only trade when not excessively choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA34 (34*7=238 bars approx), volume MA (20), Chop (14)
    # Using 1w EMA34: 34 weeks * 7 days = 238 days minimum
    start_idx = max(238, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above H3 + 1w uptrend + volume spike + not choppy
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike[i] and \
                         not_choppy[i]
            # Short: price closes below L3 + 1w downtrend + volume spike + not choppy
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_spike[i] and \
                          not_choppy[i]
            
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
            # Exit: price closes below L3 OR 1w trend turns down
            if (close[i] < camarilla_l3_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above H3 OR 1w trend turns up
            if (close[i] > camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0