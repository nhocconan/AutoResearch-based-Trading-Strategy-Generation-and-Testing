#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation.
# Long: Close breaks above Camarilla R3 AND 1d ADX > 25 (trending) AND volume > 1.8x 20-period MA
# Short: Close breaks below Camarilla S3 AND 1d ADX > 25 (trending) AND volume > 1.8x 20-period MA
# Exit: Opposite Camarilla breakout or ADX < 20 (range) or volume drops below 1.2x MA.
# Uses proven Camarilla pivot structure with tight volume confirmation (1.8x) to reduce false breakouts.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull via long signals and bear via short signals when aligned with 1d trend.

name = "4h_Camarilla_R3S3_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low), 
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)), 
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # But we need intraday levels based on previous day's range
    # For 4h timeframe, we use previous 1d OHLC to calculate levels
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = prev_1d_close + 1.125 * (prev_1d_high - prev_1d_low)
    camarilla_s3 = prev_1d_close - 1.125 * (prev_1d_high - prev_1d_low)
    camarilla_r4 = prev_1d_close + 1.5 * (prev_1d_high - prev_1d_low)
    camarilla_s4 = prev_1d_close - 1.5 * (prev_1d_high - prev_1d_low)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume regime: current 4h volume > 1.8x 20-period MA (tighter confirmation)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    volume_normal = volume > (1.2 * vol_ma_20)  # For exit condition
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        vol_normal = volume_normal[i]
        
        # Determine trend regime
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND trending AND volume spike
            if close_val > r3_aligned[i] and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND trending AND volume spike
            elif close_val < s3_aligned[i] and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S3 OR trend weakens (ADX < 20) OR volume drops below normal
            if close_val < s3_aligned[i] or is_ranging or not vol_normal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R3 OR trend weakens (ADX < 20) OR volume drops below normal
            if close_val > r3_aligned[i] or is_ranging or not vol_normal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals