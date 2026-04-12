#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volume confirmation and chop regime filter
    # Donchian(20) breakout captures institutional participation in ranging/weak trending markets
    # Volume > 2x 20-period MA confirms breakout validity
    # Chop > 61.8 ensures we trade in ranging markets where breakouts often revert
    # Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume MA and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for chop regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation for 1d
    tr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14) for 1d
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i])
    
    # Highest high and lowest low over 14 periods for 1d
    highest_high_1d = np.full(len(close_1d), np.nan)
    lowest_low_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        highest_high_1d[i] = np.max(high_1d[i-14:i])
        lowest_low_1d[i] = np.min(low_1d[i-14:i])
    
    # Chop = log10(sum(atr(14))/abs(highest_high - lowest_low)) * log10(14) * 100
    chop_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if highest_high_1d[i] != lowest_low_1d[i] and atr_1d[i] > 0:
            sum_atr = np.sum(atr_1d[i-14:i])
            chop_1d[i] = np.log10(sum_atr / abs(highest_high_1d[i] - lowest_low_1d[i])) * np.log10(14) * 100
        else:
            chop_1d[i] = 50.0
    
    # Align chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian(20) on 12h timeframe
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 2x 20-period MA on 12h
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime: Chop > 61.8 = ranging (good for breakout fade)
        ranging_market = chop_1d_aligned[i] > 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Entry conditions: breakout with volume confirmation in ranging market
        long_entry = breakout_up and (vol_ratio[i] > 2.0) and ranging_market
        short_entry = breakout_down and (vol_ratio[i] > 2.0) and ranging_market
        
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

name = "12h_1d_donchian_breakout_vol_chop_v1"
timeframe = "12h"
leverage = 1.0