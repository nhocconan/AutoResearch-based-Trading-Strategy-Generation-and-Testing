#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike filter (volume > 2.0x 20-period median) and 1d ADX filter (ADX > 25 for trending)
# Long when price > Camarilla R3 AND 1d volume > 2.0x 20d median volume AND 1d ADX > 25
# Short when price < Camarilla S3 AND 1d volume > 2.0x 20d median volume AND 1d ADX > 25
# Exit when price crosses Camarilla pivot point (mean reversion)
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Combines pivot-based price structure with volume confirmation and trend strength filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels, volume filter, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla pivot levels (R3, S3, pivot) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r3_1d = pp_1d + range_1d * 1.1 / 4.0
    s3_1d = pp_1d - range_1d * 1.1 / 4.0
    
    # === 1d Indicators: Volume median for scaling ===
    volume_1d = df_1d['volume'].values
    volume_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # === 1d Indicators: ADX (14-period) for trend strength ===
    # True Range (TR)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement (DM)
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr_14 + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    volume_median_aligned = align_htf_to_ltf(prices, df_1d, volume_median_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Align 1d volume for volume confirmation
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 14)  # 1d Camarilla needs range, 1d volume median, 1d ADX
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(volume_median_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        pp = pp_aligned[i]
        vol_median = volume_median_aligned[i]
        adx = adx_aligned[i]
        vol_1d = volume_1d_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Camarilla pivot point (mean reversion)
            if price < pp:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Camarilla pivot point (mean reversion)
            if price > pp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: current 1d volume > 2.0x 20d median volume (stricter filter to reduce trades)
            vol_threshold = vol_median * 2.0
            vol_confirm = vol_1d > vol_threshold
            
            # Trend filter: ADX > 25 indicates strong trend
            trend_confirm = adx > 25
            
            # LONG CONDITIONS
            # Price breaks above Camarilla R3 AND volume confirmation AND trend confirmation
            if price > r3 and vol_confirm and trend_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Camarilla S3 AND volume confirmation AND trend confirmation
            elif price < s3 and vol_confirm and trend_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Camarilla_R3S3_1dVolumeSpike2.0x_ADX25_v1"
timeframe = "12h"
leverage = 1.0