#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with 1-week EMA trend filter and volume confirmation
# Long when close > Donchian upper(20) + close > 1-week EMA50 + volume spike
# Short when close < Donchian lower(20) + close < 1-week EMA50 + volume spike
# Exit when price crosses 10-period EMA on 1d
# Designed for low trade frequency (~15-25/year) to minimize fee drain in 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to lower timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 10-period EMA on 1d for exit
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_10_aligned[i]) or 
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
        ema50 = ema_50_aligned[i]
        ema10 = ema_10_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price > Donchian upper + uptrend + volume spike
            if price > upper and price > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower + downtrend + volume spike
            elif price < lower and price < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses 10-period EMA
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price drops below 10 EMA
                if price < ema10:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above 10 EMA
                if price > ema10:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0