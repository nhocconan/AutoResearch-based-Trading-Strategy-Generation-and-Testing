#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel (20) breakout with volume confirmation and 1-week EMA50 trend filter.
# Long when price breaks above upper band + volume spike + price > weekly EMA50
# Short when price breaks below lower band + volume spike + price < weekly EMA50
# Exit when price crosses back through the middle band or volume drops below 70% of average.
# Designed for 1d timeframe with 1h trend filter to reduce whipsaw and improve trend alignment.
# Target: 15-25 trades/year to minimize fee drag and capture major trend moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
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
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.7 * 20-day average
        vol_spike = vol > 1.7 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume spike + price > weekly EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume spike + price < weekly EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle band or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle band or volume dries up
                if price < middle or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle band or volume dries up
                if price > middle or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_Volume_1wEMA50"
timeframe = "1d"
leverage = 1.0