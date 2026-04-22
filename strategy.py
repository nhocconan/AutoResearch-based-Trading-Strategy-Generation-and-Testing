#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume spike
# Long when price breaks above 20-period high in uptrend (close > 1d EMA34) with volume spike (>2x 20-period avg)
# Short when price breaks below 20-period low in downtrend (close < 1d EMA34) with volume spike
# Exit when price returns to the 20-period midpoint or trend reverses
# Designed for low trade frequency (~20-40/year) to minimize fee drain. Works in bull/bear by
# combining structure-based breakouts with trend filtering and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period high/low for Donchian channel
    high = prices['high'].values
    low = prices['low'].values
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(mid_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        high_val = high_20[i]
        low_val = low_20[i]
        mid_val = mid_20[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above 20-period high + uptrend + volume spike
            if price > high_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-period low + downtrend + volume spike
            elif price < low_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to midpoint or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to midpoint or trend turns down
                if price <= mid_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to midpoint or trend turns up
                if price >= mid_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0