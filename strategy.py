#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Donchian breakouts capture momentum moves in both bull and bear markets.
# 1d EMA34 provides higher timeframe trend direction: only take longs when price > 1d EMA34, shorts when price < 1d EMA34.
# Volume confirmation requires current volume > 1.3x 20-period average to avoid false signals.
# Designed to work in both bull and bear markets by aligning with trend via 1d EMA34 filter.
# Targets 15-30 trades/year with strict entry conditions to minimize fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d data
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 12h data (20-period)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_12h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_12h, lowest_low)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_channel = highest_high_aligned[i]
        lower_channel = lowest_low_aligned[i]
        ema_val = ema_1d_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_spike = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above upper Donchian + uptrend + volume spike
            if price > upper_channel and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below lower Donchian + downtrend + volume spike
            elif price < lower_channel and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below lower Donchian or trend breaks
                if price < lower_channel or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above upper Donchian or trend breaks
                if price > upper_channel or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0