#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian(20) breakout + daily volume confirmation.
# Uses weekly Donchian channels to capture long-term trends. 
# Long when price breaks above weekly Donchian high with volume > 1.5x daily average.
# Short when price breaks below weekly Donchian low with volume > 1.5x daily average.
# Exits when price returns to weekly Donchian midpoint.
# Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    weekly_mid = (weekly_high + weekly_low) / 2
    
    # Align weekly Donchian to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Calculate daily average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_mid_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        upper = weekly_high_aligned[i]
        lower = weekly_low_aligned[i]
        mid = weekly_mid_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirmed = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Enter long on weekly Donchian breakout with volume confirmation
            if price > upper and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short on weekly Donchian breakdown with volume confirmation
            elif price < lower and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to weekly Donchian midpoint
            if position == 1 and price <= mid:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price >= mid:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0