#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above upper Donchian + price > 1w EMA200 + volume spike
# Short when price breaks below lower Donchian + price < 1w EMA200 + volume spike
# Exit when price returns to opposite Donchian level or trend reverses
# Designed for low trade frequency (~10-20/year) with strong edge in both bull and bear markets
# Donchian channels provide dynamic support/resistance, EMA200 filters trend direction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels from daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: 20-day high
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-day low  
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 200-period EMA on weekly data for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema_val = ema_200_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + uptrend + volume spike
            if price > upper_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + downtrend + volume spike
            elif price < lower_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to lower Donchian or trend turns down
                if price <= lower_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to upper Donchian or trend turns up
                if price >= upper_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0