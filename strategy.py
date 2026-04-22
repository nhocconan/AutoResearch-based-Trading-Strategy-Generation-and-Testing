#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA40 trend filter and volume confirmation
# Long when price breaks above 20-day high + price > 1w EMA40 + volume spike
# Short when price breaks below 20-day low + price < 1w EMA40 + volume spike
# Exit when price returns to opposite Donchian level or trend reverses
# Designed for low trade frequency (~10-25/year) with strong edge in both bull and bear markets
# Uses Donchian channels for institutional breakout levels and weekly EMA for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels (using previous day's data to avoid look-ahead)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (no additional shift needed as already shifted)
    donchian_high = high_max_20
    donchian_low = low_min_20
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA40 for trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_40_aligned[i]) or 
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
        ema_val = ema_40_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above 20-day high + uptrend + volume spike
            if price > upper and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-day low + downtrend + volume spike
            elif price < lower and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to 20-day low or trend turns down
                if price < lower or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to 20-day high or trend turns up
                if price > upper or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA40_Volume"
timeframe = "1d"
leverage = 1.0