#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with volume spike and 1d EMA200 trend filter
# Long when price breaks above Donchian upper band + volume spike + price > 1d EMA200
# Short when price breaks below Donchian lower band + volume spike + price < 1d EMA200
# Exit when price crosses back through Donchian middle band or volume dries up
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets
# Target: 15-25 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        ema200 = ema200_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume spike + price > EMA200
            if price > upper and vol_spike and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume spike + price < EMA200
            elif price < lower and vol_spike and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle band or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle band or volume dries up
                if price < middle or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle band or volume dries up
                if price > middle or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_EMA200_Volume"
timeframe = "4h"
leverage = 1.0