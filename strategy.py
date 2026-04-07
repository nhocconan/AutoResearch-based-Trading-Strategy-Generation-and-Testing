#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter + 1-day Donchian(20) breakout with volume confirmation
# Long when 12h Choppiness > 61.8 (ranging market) and price breaks above daily Donchian upper(20) with volume
# Short when 12h Choppiness > 61.8 (ranging market) and price breaks below daily Donchian lower(20) with volume
# Exit when price crosses opposite Donchian level or Choppiness drops below 38.2 (trending market)
# Volume confirmation: current volume > 1.5 * average volume of last 20 periods
# Position size: 0.25 (25% of capital)
# Choppiness filter prevents trading in strong trends where breakouts fail, focusing on ranging markets
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag

name = "12h_chop_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: 100 * log10(sum(TR14) / (14 * (HH14 - LL14))) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high_14 - lowest_low_14
    # Avoid division by zero
    hh_ll_diff = np.where(hh_ll_diff == 0, 1e-10, hh_ll_diff)
    chop = 100 * np.log10(tr_sum / (14 * hh_ll_diff)) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian lower(20) OR Choppiness < 38.2 (trending)
            if close[i] < donch_low_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian upper(20) OR Choppiness < 38.2 (trending)
            if close[i] > donch_high_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Chop filter: only trade when market is ranging (Chop > 61.8)
            ranging = chop_aligned[i] > 61.8
            
            # Volume confirmation: current volume > 1.5 * average volume
            volume_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in ranging market with volume
            if close[i] > donch_high_aligned[i] and ranging and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in ranging market with volume
            elif close[i] < donch_low_aligned[i] and ranging and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals