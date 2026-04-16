#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1d Donchian(20) breakout with 1d volume confirmation and 1w EMA50 trend filter.
# Long when price > 1d upper Donchian, 1d volume > 1.8x 20-period median volume, and 1w close > 1w EMA50.
# Short when price < 1d lower Donchian, same volume condition, and 1w close < 1w EMA50.
# Exit when price crosses the 1d middle Donchian band.
# Uses discrete position size 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# This targets BTC/ETH with a weekly trend filter to avoid counter-trend entries and reduce overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (optional, but can be used if needed)
    # For 1d timeframe, we can use daily session or remove entirely
    # Since we're on 1d, we'll skip session filter to avoid over-filtering
    
    # Get 1d data once before loop for Donchian levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian channel (20-period) and volume median ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate 1d volume median (20-period)
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA50 for trend ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1d)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50)  # Donchian(20), EMA50(1w)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_1d = vol_1d_aligned[i]
        ema_50_1w = ema_50_1w_aligned[i]
        
        # Get aligned 1d close for proper trend comparison
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_50_1w
        trend_down = close_1d_aligned[i] < ema_50_1w
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 1d volume > 1.8x median volume
            volume_spike = vol_1d > (vol_median * 1.8)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND 1w uptrend
            if price > upper and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND 1w downtrend
            elif price < lower and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_Donchian20_1dVolumeSpike1.8x_1wEMA50_v1"
timeframe = "1d"
leverage = 1.0