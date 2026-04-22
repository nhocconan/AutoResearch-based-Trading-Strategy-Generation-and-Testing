#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume spike + weekly trend filter
# Long when price breaks above Donchian upper band (20-period high) + volume spike + price > weekly EMA50
# Short when price breaks below Donchian lower band (20-period low) + volume spike + price < weekly EMA50
# Exit when price crosses back through the 20-period middle band or volume dries up
# Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# Target: 15-30 trades/year to avoid fee drag on 12h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels: upper = 20-period high, lower = 20-period low, middle = average
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(ema50_weekly_aligned[i]) or 
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
        weekly_ema = ema50_weekly_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume spike + price > weekly EMA50
            if price > upper and vol_spike and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume spike + price < weekly EMA50
            elif price < lower and vol_spike and price < weekly_ema:
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

name = "12h_Donchian20_WeeklyEMA50_Volume_Spike"
timeframe = "12h"
leverage = 1.0