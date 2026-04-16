#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
# Long when price breaks above Donchian upper AND price > 1d EMA50 AND volume > 1.5x avg volume.
# Short when price breaks below Donchian lower AND price < 1d EMA50 AND volume > 1.5x avg volume.
# Exit when price crosses Donchian midpoint OR price crosses 1d EMA50.
# Uses discrete position size 0.25. Donchian provides structure, EMA50 filters trend, volume confirms.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns) with trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Donchian(20) channels ===
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (upper + lower) / 2
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper_4h = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume confirmation: 1.5x 20-period average volume ===
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # EMA50 and Donchian need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        vol_thresh = vol_threshold[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian middle OR price < EMA50 (trend break)
            if (price < middle) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian middle OR price > EMA50 (trend break)
            if (price > middle) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian upper AND price > EMA50 AND volume > threshold
            if (price > upper) and (price > ema50) and (vol > vol_thresh):
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian lower AND price < EMA50 AND volume > threshold
            elif (price < lower) and (price < ema50) and (vol > vol_thresh):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0