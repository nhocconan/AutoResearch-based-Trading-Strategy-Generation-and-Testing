#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + weekly EMA200 trend + volume spike
# Long when price breaks above Donchian(20) high AND weekly EMA200 trending up AND volume spike
# Short when price breaks below Donchian(20) low AND weekly EMA200 trending down AND volume spike
# Exit when price crosses back through Donchian(20) midline
# Designed for low trade frequency (~15-30/year) with edge in trending markets
# Works in both bull (strong uptrends) and bear (strong downtrends) by following weekly trend

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend direction
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_prev = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().shift(1).values
    
    # Align weekly EMA200 to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_1w_prev_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w_prev)
    
    # Calculate Donchian(20) levels on 12h data
    high_20 = pd.Series(prices['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low']).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(ema200_1w_prev_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200_now = ema200_1w_aligned[i]
        ema200_prev = ema200_1w_prev_aligned[i]
        
        # Weekly trend: up if current EMA200 > previous EMA200
        weekly_up = ema200_now > ema200_prev
        weekly_down = ema200_now < ema200_prev
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, weekly trend up, volume spike
            if price > high_20[i] and weekly_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, weekly trend down, volume spike
            elif price < low_20[i] and weekly_down and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Donchian midline
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below midline
                if price < mid_20[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above midline
                if price > mid_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA200_Trend_Volume"
timeframe = "12h"
leverage = 1.0