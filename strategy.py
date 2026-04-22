#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above upper band + price > weekly EMA34 + volume spike
# Short when price breaks below lower band + price < weekly EMA34 + volume spike
# Exit when price returns to middle band or trend reverses
# Designed for low trade frequency (~10-20/year) with strong edge in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate Donchian channels (20-period)
    upper_band = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2
    
    # Align Donchian bands to daily timeframe (previous day's levels)
    upper_band_aligned = align_htf_to_ltf(prices, df_daily, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_daily, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_daily, middle_band)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        middle_val = middle_band_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + uptrend + volume spike
            if price > upper_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + downtrend + volume spike
            elif price < lower_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to middle band or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to middle band or trend turns down
                if price <= middle_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to middle band or trend turns up
                if price >= middle_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian20_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0