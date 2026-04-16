#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 1.3x 20-period average AND 1w ADX > 20.
# Short when price breaks below Camarilla S3 AND 1d volume > 1.3x 20-period average AND 1w ADX > 20.
# Exit when price reaches Camarilla R4/S4 (profit target) or crosses R3/S3 in opposite direction (stop).
# Uses discrete position size 0.25. Designed to capture institutional breakout/continuation moves.
# Target: 80-180 total trades over 4 years (20-45/year) balancing edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on prior 6h bar) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate prior 6h bar's Camarilla levels
    # Using prior bar to avoid look-ahead: R3/S3 based on bar i-1
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h = np.roll(close_6h, 1)
    prev_high_6h[0] = np.nan
    prev_low_6h[0] = np.nan
    prev_close_6h[0] = np.nan
    
    pivot_6h = (prev_high_6h + prev_low_6h + prev_close_6h) / 3
    range_6h = prev_high_6h - prev_low_6h
    
    # Camarilla levels
    r3_6h = pivot_6h + range_6h * 1.1 / 4
    s3_6h = pivot_6h - range_6h * 1.1 / 4
    r4_6h = pivot_6h + range_6h * 1.1 / 2
    s4_6h = pivot_6h - range_6h * 1.1 / 2
    
    # Align 6h Camarilla levels to 6h primary timeframe (no shift needed as based on prior bar)
    r3_6h_aligned = r3_6h
    s3_6h_aligned = s3_6h
    r4_6h_aligned = r4_6h
    s4_6h_aligned = s4_6h
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # === 1w Indicators: ADX > 20 (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1w).diff()
    dm_minus = pd.Series(low_1w).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    strong_trend = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3_6h_aligned[i]) or np.isnan(s3_6h_aligned[i]) or np.isnan(r4_6h_aligned[i]) or np.isnan(s4_6h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reaches R4 (profit target) or crosses below R3 (stop)
            if price >= r4_6h_aligned[i] or price < r3_6h_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reaches S4 (profit target) or crosses above S3 (stop)
            if price <= s4_6h_aligned[i] or price > s3_6h_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 AND volume spike AND strong trending market
            if price > r3_6h_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below S3 AND volume spike AND strong trending market
            elif price < s3_6h_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dVolumeSpike_1wADX_V1"
timeframe = "6h"
leverage = 1.0