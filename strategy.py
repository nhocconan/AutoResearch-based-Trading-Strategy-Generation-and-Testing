#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Donchian breakout + volume confirmation + chop filter
# Long when price breaks above 1d Donchian Upper (20) and volume > 1.5x avg volume
# Short when price breaks below 1d Donchian Lower (20) and volume > 1.5x avg volume
# Chop filter: only trade when CHOP(14) > 61.8 (ranging market) to avoid false breakouts in trends
# Uses daily timeframe for structure to avoid whipsaws, 12h for execution timing
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Choppy Index (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(tr_sum / (hh14 - ll14)) / np.log10(14)
    
    # Align 1d indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume + chop filter
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        ranging_market = chop_aligned[i] > 61.8  # Chop > 61.8 = ranging
        
        long_breakout = close[i] > donch_high_aligned[i]
        short_breakout = close[i] < donch_low_aligned[i]
        
        long_entry = long_breakout and volume_spike and ranging_market
        short_entry = short_breakout and volume_spike and ranging_market
        
        # Exit conditions: opposite Donchian breakout
        exit_long = position == 1 and close[i] < donch_low_aligned[i]
        exit_short = position == -1 and close[i] > donch_high_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_volume_chop"
timeframe = "12h"
leverage = 1.0