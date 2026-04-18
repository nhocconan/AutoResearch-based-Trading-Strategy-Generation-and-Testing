#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and Trend Filter
Hypothesis: Weekly Donchian channels capture long-term breakouts. Price breaking
above/below the 20-week high/low with volume confirmation and aligned with
daily trend (via 1d EMA34) captures institutional moves. Works in both bull
and bear markets by filtering counter-trend trades. Target: 10-20 trades/year
to minimize fee drift while capturing strong directional moves.
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
    
    # Get weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (wait for weekly bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume, in uptrend
            if price > dh and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume, in downtrend
            elif price < dl and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns below weekly Donchian high or trend weakens
            if price < dh or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns above weekly Donchian low or trend weakens
            if price > dl or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0