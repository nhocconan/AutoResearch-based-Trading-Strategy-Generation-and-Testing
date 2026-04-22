#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation
# Long when price breaks above 1h Donchian upper + 4h price > 4h EMA50 + volume spike
# Short when price breaks below 1h Donchian lower + 4h price < 4h EMA50 + volume spike
# Exit when price returns to 1h Donchian middle or 4h trend reverses
# Uses 4h for signal direction (trend), 1h only for entry timing to control trade frequency
# Designed for 15-30 trades/year with strong edge in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter and Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper (20-period high)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian middle (average of upper and lower)
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_4h = ema_50_4h_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + 4h uptrend + volume spike
            if price > donch_high_val and price > ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Donchian low + 4h downtrend + volume spike
            elif price < donch_low_val and price < ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian middle or 4h trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to Donchian middle or 4h trend turns down
                if price <= donch_mid_val or price < ema_4h:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to Donchian middle or 4h trend turns up
                if price >= donch_mid_val or price > ema_4h:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0