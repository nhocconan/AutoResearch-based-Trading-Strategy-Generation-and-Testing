#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
    # Uses 1d ATR for volatility normalization and chop calculation
    # Volume > 1.5x 20-period MA confirms institutional participation
    # Chop > 61.8 ensures ranging markets where mean reversion works on breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 25-40 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14)
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i])
    
    # Align ATR to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian channels (20-period) on 4h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    # Chop regime filter: CHOP > 61.8 = ranging market
    # Calculate ATR(14) on 4h
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr_4h = np.full(n, np.nan)
    for i in range(14, n):
        atr_4h[i] = np.mean(tr[i-14:i])
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = np.full(n, np.nan)
    lowest_low_14 = np.full(n, np.nan)
    for i in range(14, n):
        highest_high_14[i] = np.max(high[i-14:i])
        lowest_low_14[i] = np.min(low[i-14:i])
    
    # Chop = log10(sum(atr(14))/abs(highest_high - lowest_low)) * log10(14) * 100
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if highest_high_14[i] != lowest_low_14[i] and atr_4h[i] > 0:
            sum_atr = np.sum(atr_4h[i-14:i])
            chop[i] = np.log10(sum_atr / abs(highest_high_14[i] - lowest_low_14[i])) * np.log10(14) * 100
        else:
            chop[i] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime: Chop > 61.8 = ranging (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Entry conditions: breakout with volume confirmation in ranging market
        long_entry = breakout_up and (vol_ratio[i] > 1.5) and ranging_market
        short_entry = breakout_down and (vol_ratio[i] > 1.5) and ranging_market
        
        # Exit conditions: price returns to midpoint of Donchian channel
        midpoint = (highest_high[i] + lowest_low[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_vol_chop_v1"
timeframe = "4h"
leverage = 1.0