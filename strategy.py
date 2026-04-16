#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price > 1d Donchian Upper AND close > 1w EMA50 (uptrend).
# Short when price < 1d Donchian Lower AND close < 1w EMA50 (downtrend).
# Exit when price crosses 1d Donchian midpoint OR trend reverses.
# Uses discrete position size 0.25. Donchian channels provide clear breakout levels.
# 1w EMA50 ensures trading only with higher timeframe trend to avoid whipsaws.
# Volume confirmation filters weak breakouts.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (capture uptrend breakouts) and bear markets (capture downtrend breakouts).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Donchian(20) channels ===
    # Donchian Upper = max(high, 20)
    # Donchian Lower = min(low, 20)
    # Donchian Middle = (Upper + Lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # === 1d Indicators: Volume confirmation (20-period volume MA ratio) ===
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / np.where(volume_ma > 0, volume_ma, 1)  # avoid division by zero
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_ratio_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol_ratio = volume_ratio_aligned[i]
        dc_upper = donchian_upper_aligned[i]
        dc_lower = donchian_lower_aligned[i]
        dc_middle = donchian_middle_aligned[i]
        ema50 = ema50_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian Middle OR trend breaks (price < EMA50)
            if (price < dc_middle) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian Middle OR trend breaks (price > EMA50)
            if (price > dc_middle) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian Upper AND volume confirmation AND uptrend (price > EMA50)
            if (price > dc_upper) and (vol_ratio > 1.5) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian Lower AND volume confirmation AND downtrend (price < EMA50)
            elif (price < dc_lower) and (vol_ratio > 1.5) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_1wEMA50_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0