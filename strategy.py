#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above upper band + price > weekly EMA50 + volume spike
# Short when price breaks below lower band + price < weekly EMA50 + volume spike
# Exit when price crosses back through middle band or trend reverses
# Designed for low trade frequency (~15-30/year) to minimize fee drain and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate daily data for Donchian channels (20-period)
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Calculate 20-period Donchian channels on daily timeframe
    # Upper band = 20-period high, Lower band = 20-period low, Middle band = average
    upper_band = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2
    
    # Align Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower_band)
    middle_aligned = align_htf_to_ltf(prices, df_daily, middle_band)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + uptrend + volume spike
            if price > upper and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + downtrend + volume spike
            elif price < lower and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle band or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle band or trend turns down
                if price < middle or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle band or trend turns up
                if price > middle or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0