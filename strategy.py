#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend + volume spike
# Long when price breaks above upper Donchian + price > EMA34 + volume spike
# Short when price breaks below lower Donchian + price < EMA34 + volume spike
# Exit when price crosses back through EMA34
# Designed for low trade frequency (~15-30/year) with edge in trending markets
# Works in both bull (strong uptrends) and bear (strong downtrends)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    
    # Upper band: highest high over last 20 periods
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d EMA to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + price > EMA34 + volume spike
            if price > upper_band[i] and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + price < EMA34 + volume spike
            elif price < lower_band[i] and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through EMA34
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below EMA34
                if price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above EMA34
                if price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0