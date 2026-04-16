#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter.
# Long when price breaks above 1d Donchian upper channel AND volume > 1.5x 20-period average AND close > 1w EMA50 (uptrend).
# Short when price breaks below 1d Donchian lower channel AND volume > 1.5x 20-period average AND close < 1w EMA50 (downtrend).
# Exit when price crosses the Donchian middle (20-period mean) or volume drops below average.
# Uses discrete position size 0.25. Donchian channels provide clear breakout levels with defined exits.
# 1w EMA50 ensures trading only with higher timeframe trend to avoid whipsaws in ranging markets.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (capture upside breakouts) and bear markets (capture downside breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels and volume average
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
    
    # === 1d Indicators: Donchian(20) channels and volume average ===
    # Donchian upper = max(high, 20)
    # Donchian lower = min(low, 20)
    # Donchian middle = (upper + lower) / 2
    high_roll_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_20
    donchian_lower = low_roll_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20-period)
    volume_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    volume_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_avg_20)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_avg_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        dc_upper = donchian_upper_aligned[i]
        dc_lower = donchian_lower_aligned[i]
        dc_middle = donchian_middle_aligned[i]
        vol_avg = volume_avg_aligned[i]
        ema50 = ema50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol > 1.5 * vol_avg
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian middle OR volume confirmation fails
            if (price < dc_middle) or (not volume_confirm):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian middle OR volume confirmation fails
            if (price > dc_middle) or (not volume_confirm):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian upper AND volume confirmation AND price > 1w EMA50 (uptrend)
            if (price > dc_upper) and volume_confirm and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian lower AND volume confirmation AND price < 1w EMA50 (downtrend)
            elif (price < dc_lower) and volume_confirm and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_1wEMA50_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0