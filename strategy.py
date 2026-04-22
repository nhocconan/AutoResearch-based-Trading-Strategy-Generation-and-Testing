#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w MA50 trend filter and volume confirmation
# Long when price breaks above 20-day high + price > weekly MA50 + volume spike
# Short when price breaks below 20-day low + price < weekly MA50 + volume spike
# Exit when price returns to opposite Donchian level or trend reverses
# Designed for low trade frequency (~10-20/year) with strong trend following in both bull and bear markets
# Weekly MA50 filters for long-term trend direction to avoid counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channel (20-period high/low)
    # Donchian High = max(high over last 20 days)
    # Donchian Low = min(low over last 20 days)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned, but keep for consistency)
    donchian_high_aligned = donchian_high  # already 1d data
    donchian_low_aligned = donchian_low    # already 1d data
    
    # Load 1w data for MA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly MA50 for trend filter
    ma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    ma_50_aligned = align_htf_to_ltf(prices, df_1w, ma_50_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ma_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ma_val = ma_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if price > donchian_high_val and price > ma_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif price < donchian_low_val and price < ma_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to Donchian low or trend turns down
                if price <= donchian_low_val or price < ma_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to Donchian high or trend turns up
                if price >= donchian_high_val or price > ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wMA50_Volume"
timeframe = "1d"
leverage = 1.0