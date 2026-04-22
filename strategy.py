#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above Donchian upper + weekly pivot bullish + volume spike
# Short when price breaks below Donchian lower + weekly pivot bearish + volume spike
# Exit when price crosses Donchian midpoint or volume drops below 80% of average
# Weekly pivot provides directional bias from higher timeframe, reducing false breakouts
# Target: 15-30 trades/year to minimize fee drag while capturing strong trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data (using prior week's data)
    # Weekly high = max of last 5 daily highs
    # Weekly low = min of last 5 daily lows
    # Weekly close = last daily close of the week
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    
    # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly bias: bullish if close > weekly_pivot, bearish if close < weekly_pivot
    weekly_bullish = close_1d > weekly_pivot
    weekly_bearish = close_1d < weekly_pivot
    
    # Align weekly bias to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    # Load 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 6h timeframe (already aligned since from 6h data)
    # But we still need to handle the alignment properly for look-ahead safety
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        mid = donchian_mid_aligned[i]
        bullish = weekly_bullish_aligned[i] > 0.5
        bearish = weekly_bearish_aligned[i] > 0.5
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + weekly bullish + volume spike
            if price > upper and bullish and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + weekly bearish + volume spike
            elif price < lower and bearish and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Donchian midpoint or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below midpoint or volume dries up
                if price < mid or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above midpoint or volume dries up
                if price > mid or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0