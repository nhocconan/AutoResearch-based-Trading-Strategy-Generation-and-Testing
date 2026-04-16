#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above 1d Donchian upper channel AND close > 1w EMA50 (uptrend) AND volume > 1.5x average volume.
# Short when price breaks below 1d Donchian lower channel AND close < 1w EMA50 (downtrend) AND volume > 1.5x average volume.
# Exit when price crosses the 1d Donchian midpoint (mean of upper/lower channel) or 1w EMA50.
# Uses discrete position size 0.25. Donchian channels provide clear structure, EMA50 filters higher timeframe trend.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (capture uptrend breakouts) and bear markets (capture downtrend breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Donchian(20) ===
    # Upper Channel = max(high, lookback=20)
    # Lower Channel = min(low, lookback=20)
    # Middle Channel = (upper + lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume: average volume over 20 periods ===
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = vol_ma20_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian middle OR price < 1w EMA50
            if (price < middle) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian middle OR price > 1w EMA50
            if (price > middle) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian upper AND price > 1w EMA50 AND volume > 1.5x average volume
            if (price > upper) and (price > ema50) and (vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian lower AND price < 1w EMA50 AND volume > 1.5x average volume
            elif (price < lower) and (price < ema50) and (vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dDonchian20_1wEMA50_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0