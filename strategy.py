#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Donchian breakouts capture trending moves, while EMA34 ensures alignment with daily trend.
# Volume spikes (>2x 20-period average) confirm institutional participation.
# Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for Donchian calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period)
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 34-period EMA on 1d close for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 12h timeframe (no additional delay needed for Donchian)
    # Since we're using 12h as primary timeframe, we need to align 1d EMA to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_channel = high_max_20[i]
        lower_channel = low_min_20[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper channel + uptrend + volume spike
            if price > upper_channel and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower channel + downtrend + volume spike
            elif price < lower_channel and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below lower channel or trend breaks
                if price < lower_channel or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above upper channel or trend breaks
                if price > upper_channel or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0