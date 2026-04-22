#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume spike and 1d EMA34 trend filter.
# Long when price breaks above upper Donchian channel + volume spike + price > 1d EMA34
# Short when price breaks below lower Donchian channel + volume spike + price < 1d EMA34
# Exit when price crosses back through the middle of the Donchian channel or volume drops below 80% of average.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 12-37 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: 20-period high
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle_20 = (upper_20 + lower_20) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or 
            np.isnan(middle_20[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_20[i]
        lower = lower_20[i]
        middle = middle_20[i]
        ema34 = ema34_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume spike + price > EMA34
            if price > upper and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume spike + price < EMA34
            elif price < lower and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle of Donchian or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle or volume dries up
                if price < middle or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle or volume dries up
                if price > middle or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Volume_EMA34"
timeframe = "12h"
leverage = 1.0