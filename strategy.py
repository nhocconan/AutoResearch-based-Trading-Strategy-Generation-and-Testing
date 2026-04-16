#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 12h Donchian(15) breakout with 1d volume confirmation (volume > 1.5x 20-period median) and 1d ADX(14) > 25 for trend filter.
# Long when price > 12h upper Donchian, 1d volume > 1.5x median volume, and 1d ADX > 25.
# Short when price < 12h lower Donchian, same volume condition, and 1d ADX > 25.
# Exit when price crosses the 12h middle Donchian band.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# This targets BTC/ETH with a 1d trend filter to avoid choppy markets and reduce overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian channel (15-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels
    upper_15 = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    lower_15 = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    middle_15 = (upper_15 + lower_15) / 2.0
    
    # Get 1d data for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Volume median (20-period) and ADX(14) ===
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume median (20-period)
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Calculate ADX(14)
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Align all indicators to primary timeframe (12h)
    upper_15_aligned = align_htf_to_ltf(prices, df_12h, upper_15)
    lower_15_aligned = align_htf_to_ltf(prices, df_12h, lower_15)
    middle_15_aligned = align_htf_to_ltf(prices, df_12h, middle_15)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx.values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(15, 20, 14)  # Donchian(15), volume median(20), ADX(14)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_15_aligned[i]) or np.isnan(lower_15_aligned[i]) or 
            np.isnan(middle_15_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_15_aligned[i]
        lower = lower_15_aligned[i]
        middle = middle_15_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_1d = vol_1d_aligned[i]
        adx_val = adx_aligned[i]
        
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
            # Volume spike filter: current 1d volume > 1.5x median volume
            volume_spike = vol_1d > (vol_median * 1.5)
            # Trend filter: 1d ADX > 25
            trend_filter = adx_val > 25
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND trend filter
            if price > upper and volume_spike and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND trend filter
            elif price < lower and volume_spike and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian15_1dVolumeSpike1.5x_1dADX25_v1"
timeframe = "12h"
leverage = 1.0