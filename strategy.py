#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and ADX regime filter.
# Long when price > 1d upper Donchian, 4h volume > 1.5x median, and 1d ADX > 25 (trending).
# Short when price < 1d lower Donchian, same volume condition, and 1d ADX > 25.
# Exit when price crosses the 1d middle Donchian band.
# Uses discrete position size 0.25. Session filter: 08-20 UTC.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data once before loop for Donchian levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian channel (20-period) and ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate ADX(14)
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    
    # +DM = high - previous_high if > previous_low - low and > 0 else 0
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # first value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Volume median for spike detection ===
    vol_4h = df_4h['volume'].values
    vol_median_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14+14+14, 20)  # Donchian(20), ADX(14+14+14), 4h volume median(20)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        adx_val = adx_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_4h = vol_4h_aligned[i]
        
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
            # Volume spike filter: current 4h volume > 1.5x median volume
            volume_spike = vol_4h > (vol_median * 1.5)
            
            # Regime filter: ADX > 25 indicates trending market
            is_trending = adx_val > 25
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND trending market
            if price > upper and volume_spike and is_trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND trending market
            elif price < lower and volume_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_4hVolumeSpike1.5x_1dADX25_v1"
timeframe = "4h"
leverage = 1.0