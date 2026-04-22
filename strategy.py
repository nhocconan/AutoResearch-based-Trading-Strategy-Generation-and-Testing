#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w EMA100 trend filter and volume spike confirmation.
# Uses weekly EMA to filter long-term trend (bullish/bearish) and enters on Donchian breakouts
# in the direction of weekly trend. Volume spike (>2x 20-period avg) confirms institutional interest.
# Designed for low trade frequency (~15-25/year) to minimize fee decay. Works in both bull and bear
# markets by following weekly trend. Donchian channels provide clear breakout levels with
# built-in volatility adjustment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for EMA100 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 100-period EMA on weekly close for long-term trend filter
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align weekly EMA to 12h timeframe (waits for weekly bar to close)
    ema_100_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Calculate Donchian channels on 12h data (20-period high/low)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high, Lower band: 20-period low
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_100_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_100_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Trend filter: price above/below weekly EMA100
        uptrend = price > ema_val
        downtrend = price < ema_val
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if price > upper and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif price < lower and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price breaks opposite Donchian band or trend fails
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or trend turns down
                if price < lower or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or trend turns up
                if price > upper or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DonchianBreakout_1wEMA100_Volume"
timeframe = "12h"
leverage = 1.0