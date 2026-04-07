#!/usr/bin/env python3
"""
1d Donchian breakout with weekly trend filter and volume confirmation.
Long when price breaks above Donchian upper band and weekly trend is up.
Short when price breaks below Donchian lower band and weekly trend is down.
Exit when price returns to Donchian middle or opposite breakout occurs.
Uses 1D Donchian channels (20) and 1W trend filter.
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period high/low) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Weekly trend filter: 1W EMA(21) slope ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    weekly_ema_prev = np.roll(weekly_ema_aligned, 1)
    weekly_ema_prev[0] = weekly_ema_aligned[0]
    weekly_trend_up = weekly_ema_aligned > weekly_ema_prev
    
    # === Volume confirmation (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_trend_up[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle line or opposite breakout
            if close[i] <= donch_mid[i] or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle line or opposite breakout
            if close[i] >= donch_mid[i] or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with weekly trend filter
            if close[i] > donch_high[i] and weekly_trend_up[i]:
                # Breakout above upper band with up trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and not weekly_trend_up[i]:
                # Breakdown below lower band with down trend -> short
                position = -1
                signals[i] = -0.25
    
    return signals