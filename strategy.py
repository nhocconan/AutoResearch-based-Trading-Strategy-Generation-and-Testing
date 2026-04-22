#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# Long when price breaks above Donchian upper band + 12h EMA50 uptrend + volume spike
# Short when price breaks below Donchian lower band + 12h EMA50 downtrend + volume spike
# Exit when price crosses Donchian middle band or trend reverses
# Donchian channels capture breakouts in trending markets, effective in both bull and bear regimes
# Designed for low trade frequency (~15-30/year) to minimize fee drain.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels (20-period)
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        ema_val = ema_50_12h_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: break above upper band + uptrend + volume spike
            if price > upper and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + downtrend + volume spike
            elif price < lower and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: cross middle band or trend reverses
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

name = "6h_Donchian20_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0