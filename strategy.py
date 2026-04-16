#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 12h volume spike (1.8x median) and 1d ADX trend filter (ADX > 25)
# Long when price > Camarilla R3 AND 12h volume > 1.8x 20-period median AND 1d ADX > 25
# Short when price < Camarilla S3 AND 12h volume > 1.8x 20-period median AND 1d ADX > 25
# Exit when price crosses Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
# Combines pivot level breakout with volume confirmation and trend regime filter for robustness in bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX (14-period) for trend regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d[0] = np.abs(high_1d[0] - low_1d[0])  # first bar
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Volume median (20-period) ===
    volume_12h = df_12h['volume'].values
    vol_median_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    vol_median_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20_12h)
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels calculated from previous 4h bar's high/low/close
    # Using 1d OHLC for Camarilla calculation (standard approach)
    camarilla_pivot = (high_4h + low_4h + close_4h) / 3.0
    camarilla_range = high_4h - low_4h
    camarilla_r3 = camarilla_pivot + (camarilla_range * 1.1 / 4.0)
    camarilla_s3 = camarilla_pivot - (camarilla_range * 1.1 / 4.0)
    camarilla_r4 = camarilla_pivot + (camarilla_range * 1.1 / 2.0)
    camarilla_s4 = camarilla_pivot - (camarilla_range * 1.1 / 2.0)
    
    # Align Camarilla levels to primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14)  # 1d ADX, 12h volume, 4h Camarilla
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_median_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.8x 20-period 12h volume median
        vol_threshold = vol_median_20_12h_aligned[i] * 1.8
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Regime filter: 1d ADX > 25 (trending market)
        regime_filter = adx_1d_aligned[i] > 25
        
        # Price levels
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot point (mean reversion)
            if price < pivot:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot point (mean reversion)
            if price > pivot:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Camarilla R3 AND volume confirmation AND trend regime
            if price > r3 and vol_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S3 AND volume confirmation AND trend regime
            elif price < s3 and vol_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Camarilla_R3S3_12hVolume1.8x_1dADX25_v1"
timeframe = "4h"
leverage = 1.0