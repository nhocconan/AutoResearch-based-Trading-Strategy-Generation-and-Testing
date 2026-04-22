#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and 12h EMA50 trend filter.
# Long when price breaks above 20-period upper band + volume spike + price > 12h EMA50
# Short when price breaks below 20-period lower band + volume spike + price < 12h EMA50
# Exit when price crosses back through the 20-period midpoint.
# Works in trending markets (breakouts with volume) and avoids chop via EMA filter.
# Target: 20-35 trades/year to minimize fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle line: average of upper and lower
    middle_20 = (upper_20 + lower_20) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(middle_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_20[i]
        lower = lower_20[i]
        middle = middle_20[i]
        ema50 = ema50_12h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume spike + price > 12h EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume spike + price < 12h EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle line
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle line
                if price < middle:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle line
                if price > middle:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_12hEMA50"
timeframe = "4h"
leverage = 1.0