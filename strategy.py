#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band + price > 1d EMA50 + volume spike
# Short when price breaks below 4h Donchian lower band + price < 1d EMA50 + volume spike
# Exit when price returns to opposite Donchian band or trend reverses
# Designed for low trade frequency (~20-40/year) with strong edge in both bull and bear markets
# Uses Donchian channels for breakout signals and EMA50 for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    
    # Donchian upper = max(high_4h, 20)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low_4h, 20)
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donch_low)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + uptrend + volume spike
            if price > upper and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + downtrend + volume spike
            elif price < lower and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian band or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to Donchian lower or trend turns down
                if price < lower or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to Donchian upper or trend turns up
                if price > upper or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0