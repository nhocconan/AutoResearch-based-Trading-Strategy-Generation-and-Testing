#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period high + price > 12h EMA50 + volume > 1.5x 20-period avg volume
# Short when price breaks below 20-period low + price < 12h EMA50 + volume > 1.5x 20-period avg volume
# Exit when price returns to opposite Donchian level or trend reverses
# Designed for low trade frequency (~20-40/year) with strong edge in both bull and bear markets
# Uses Donchian channels for volatility-based breakouts and EMA50 for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period high/low for Donchian channels (on primary 4h timeframe)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_val = ema_50_12h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average volume
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume confirmation
            if price > upper and price > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume confirmation
            elif price < lower and price < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to Donchian low or trend turns down
                if price < lower or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to Donchian high or trend turns up
                if price > upper or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0