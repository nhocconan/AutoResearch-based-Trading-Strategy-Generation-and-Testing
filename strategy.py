#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot levels (R1, S1) from daily timeframe with 4h volume confirmation and ADX trend filter.
# Long when 4h close crosses above daily S1 pivot with volume > 1.5x 20-period median and ADX > 25 (trending market).
# Short when 4h close crosses below daily R1 pivot with same volume and ADX conditions.
# Exit when price returns to daily close (mean reversion to midpoint) or opposite pivot is touched.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# This strategy works in both bull and bear markets by using ADX to filter ranging conditions and Camarilla levels
# as dynamic support/resistance that adapt to volatility, avoiding false breakouts in low-volume environments.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for volume median
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: volume median ===
    vol_4h = df_4h['volume'].values
    vol_median_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla pivot levels (R1, S1) and ADX ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = close_1d + (daily_range * 1.1 / 12)
    s1 = close_1d - (daily_range * 1.1 / 12)
    daily_close = close_1d  # Pivot point for exit
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (14-period)
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align all indicators to primary timeframe (4h)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 50)  # volume median(20), ADX components
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_median_aligned[i]) or np.isnan(vol_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(daily_close_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        vol_median = vol_median_aligned[i]
        vol_4h = vol_4h_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        daily_close_val = daily_close_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        prev_price = close[i-1]
        
        # Volume spike filter: current 4h volume > 1.5x median volume
        volume_spike = vol_4h > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price returns to daily close (mean reversion)
            if price <= daily_close_val:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price returns to daily close (mean reversion)
            if price >= daily_close_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # ADX filter: trending market (ADX > 25)
            trending = adx_val > 25
            
            # LONG CONDITIONS
            # Price crosses above S1 with volume spike and trending market
            if prev_price <= s1 and price > s1 and volume_spike and trending:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Price crosses below R1 with volume spike and trending market
            elif prev_price >= r1 and price < r1 and volume_spike and trending:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R1S1_4hVolumeSpike1.5x_1dADX25_v1"
timeframe = "4h"
leverage = 1.0