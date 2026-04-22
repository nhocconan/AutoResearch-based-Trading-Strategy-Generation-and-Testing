# [Experiment 73642] 12h Donchian Breakout with Volume Spike and 1d EMA Trend Filter
# Hypothesis: Breakouts above 12h Donchian high(20) or below low(20) with volume confirmation and 1d EMA50 trend filter
# work in both bull and bear markets by capturing momentum bursts aligned with daily trend.
# Exit when price returns to the midpoint of the Donchian channel, reducing false breakouts.
# Target: 12-37 trades/year to minimize fee drag on 12h timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + volume spike + price > EMA50
            if price > upper and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + volume spike + price < EMA50
            elif price < lower and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian midpoint
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below midpoint
                if price < mid:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above midpoint
                if price > mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0