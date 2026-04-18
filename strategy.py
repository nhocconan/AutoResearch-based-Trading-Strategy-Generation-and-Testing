#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts with weekly EMA(55) trend filter and volume spikes capture strong trends while avoiding chop. Weekly trend filter reduces false signals in ranging markets. Volume confirmation ensures institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear regimes.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(55) for trend filter
    ema_55_1w = pd.Series(df_1w['close']).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_55_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_55_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    # Use rolling window on daily high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # enough for Donchian(20) and weekly EMA(55)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_55_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        weekly_trend = ema_55_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and price above weekly EMA
            if price > upper and vol_spike and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and price below weekly EMA
            elif price < lower and vol_spike and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Stay long until price breaks below lower Donchian or weekly trend fails
            if price < lower or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stay short until price breaks above upper Donchian or weekly trend fails
            if price > upper or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0