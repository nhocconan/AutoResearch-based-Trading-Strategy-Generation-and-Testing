#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high + price > weekly EMA50 + volume spike
# Short when price breaks below 20-day low + price < weekly EMA50 + volume spike
# Exit when price returns to opposite Donchian level or trend reverses
# Designed for low trade frequency (~10-25/year) with strong edge in both bull and bear markets
# Uses Donchian channels for breakout signals and EMA50 for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels from daily data
    # Upper band = 20-day high
    # Lower band = 20-day low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe (no additional delay needed)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_band = donchian_upper_aligned[i]
        lower_band = donchian_lower_aligned[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + uptrend + volume spike
            if price > upper_band and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + downtrend + volume spike
            elif price < lower_band and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to lower band or trend turns down
                if price <= lower_band or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to upper band or trend turns up
                if price >= upper_band or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0