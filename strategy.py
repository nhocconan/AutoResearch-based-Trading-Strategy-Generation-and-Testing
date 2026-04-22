#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d EMA50 trend filter
# Long when price breaks above 12h Donchian upper (20) + volume spike + price > 1d EMA50
# Short when price breaks below 12h Donchian lower (20) + volume spike + price < 1d EMA50
# Exit when price crosses back through 12h Donchian mid (10) or volume dries up
# Target: 15-30 trades/year to avoid fee drag, works in bull (breakouts) and bear (breakdowns)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume spike filter (20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 12h Donchian channels (20-period)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    mid_20_12h = (high_20_12h + low_20_12h) / 2.0
    
    # Align to lower timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    mid_20_aligned = align_htf_to_ltf(prices, df_12h, mid_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        vol_ma = vol_ma_aligned[i]
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        mid = mid_20_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + volume spike + price > EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + volume spike + price < EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Donchian mid or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below mid or volume dries up
                if price < mid or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above mid or volume dries up
                if price > mid or vol < 0.5 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0