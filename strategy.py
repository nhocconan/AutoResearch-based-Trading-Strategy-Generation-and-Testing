#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume spike + 1d EMA50 trend filter
# Long when price breaks above upper Donchian + volume spike + price > 1d EMA50
# Short when price breaks below lower Donchian + volume spike + price < 1d EMA50
# Exit when price crosses back through the opposite Donchian band or volume dries up
# Designed for 12h timeframe to capture medium-term trends with low trade frequency (<30/year)
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on daily data
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_dc = upper_dc_aligned[i]
        lower_dc = lower_dc_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume spike + price > EMA50
            if price > upper_dc and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume spike + price < EMA50
            elif price < lower_dc and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through opposite Donchian or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below lower Donchian or volume dries up
                if price < lower_dc or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above upper Donchian or volume dries up
                if price > upper_dc or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0