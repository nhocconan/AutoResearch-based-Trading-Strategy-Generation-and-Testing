#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day/1-week Choppiness Index regime filter + Donchian breakout.
# Uses Choppiness Index to identify trending vs ranging markets: >61.8 = range (mean revert),
# <38.2 = trend (trend follow). In trending regimes, break Donchian(20) for momentum entries.
# In ranging regimes, fade at Donchian bands for mean reversion. Volume confirmation filters noise.
# Designed for 12-30 trades/year to minimize fee drag while adapting to market regimes.
# Works in bull/bear markets by switching between trend-following and mean-reversion based on volatility regime.

name = "12h_1d_1w_chop_donchian_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = np.zeros_like(tr)
    for i in range(13, len(tr)):
        atr_14[i] = np.nansum(tr[i-13:i+1])
    atr_14[:13] = np.nan
    
    # Sum of ATR(14) over 14 periods
    atr_sum_14 = np.zeros_like(tr)
    for i in range(26, len(tr)):  # 13 + 13
        atr_sum_14[i] = np.nansum(atr_14[i-13:i+1])
    atr_sum_14[:26] = np.nan
    
    # Choppiness Index: 100 * log10(ATR_sum / (ATR * period)) / log10(period)
    chop = 100 * np.log10(atr_sum_14 / (atr_14 * 14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((atr_14 == 0) | np.isnan(atr_14) | np.isnan(atr_sum_14), 50.0, chop)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian Upper (20-period high)
    donch_high_20 = np.full_like(high_1w, np.nan)
    for i in range(19, len(high_1w)):
        donch_high_20[i] = np.max(high_1w[i-19:i+1])
    
    # Donchian Lower (20-period low)
    donch_low_20 = np.full_like(low_1w, np.nan)
    for i in range(19, len(low_1w)):
        donch_low_20[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.zeros_like(volume_1d, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_avg_20[:19] = np.nan
    
    # Align all indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Regime filters based on Choppiness Index
        is_ranging = chop_aligned[i] > 61.8  # Range: mean revert
        is_trending = chop_aligned[i] < 38.2  # Trend: trend follow
        
        if is_ranging:
            # In ranging markets: fade at Donchian bands (mean reversion)
            fade_long = low[i] <= donch_low_aligned[i] and vol_filter
            fade_short = high[i] >= donch_high_aligned[i] and vol_filter
            
            # Exit when price returns to opposite Donchian band
            exit_long = high[i] >= donch_high_aligned[i] if position == 1 else False
            exit_short = low[i] <= donch_low_aligned[i] if position == -1 else False
            
            # Priority: fade > hold
            if fade_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif fade_short and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
                
        elif is_trending:
            # In trending markets: break Donchian bands (momentum)
            breakout_long = high[i] >= donch_high_aligned[i] and vol_filter
            breakout_short = low[i] <= donch_low_aligned[i] and vol_filter
            
            # Exit when price returns to opposite Donchian band
            exit_long = low[i] <= donch_low_aligned[i] if position == 1 else False
            exit_short = high[i] >= donch_high_aligned[i] if position == -1 else False
            
            # Priority: breakout > hold
            if breakout_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif breakout_short and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Choppy transition zone (38.2-61.8): stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals