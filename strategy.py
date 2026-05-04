#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h ADX trend filter and volume confirmation
# Uses 4h ADX > 25 to confirm trending markets (avoids whipsaws in ranging markets)
# Camarilla R3/S3 from prior 1d session provide institutional breakout levels
# Volume confirmation (>1.5x 20 EMA) filters low-participation false breakouts
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in both bull and bear: ADX ensures we only trade strong trends, Camarilla provides precise entry/exit levels.

name = "1h_Camarilla_R3S3_4hADX_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data for Camarilla levels (daily session)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_1d = (high_1d + low_1d + close_1d_vals) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R3, S3 levels (most significant for breakouts)
    camarilla_r3 = close_1d_vals + 1.1 * range_1d / 2.0
    camarilla_s3 = close_1d_vals - 1.1 * range_1d / 2.0
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h ADX(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di_14 = np.where(atr_14 != 0, (plus_dm_14 / atr_14) * 100, 0)
    minus_di_14 = np.where(atr_14 != 0, (minus_dm_14 / atr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  np.abs((plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100, 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + ADX > 25 (trending) + volume spike
            if close[i] > r3_aligned[i] and adx_aligned[i] > 25 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + ADX > 25 (trending) + volume spike
            elif close[i] < s3_aligned[i] and adx_aligned[i] > 25 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H4/L4 midpoint OR ADX < 20 (trend weakening) OR weak volume
            camarilla_h4 = close_1d_vals + 1.1 * range_1d / 4.0
            camarilla_l4 = close_1d_vals - 1.1 * range_1d / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if (close[i] < midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla H4/L4 midpoint OR ADX < 20 (trend weakening) OR weak volume
            camarilla_h4 = close_1d_vals + 1.1 * range_1d / 4.0
            camarilla_l4 = close_1d_vals - 1.1 * range_1d / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if (close[i] > midpoint or 
                adx_aligned[i] < 20 or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals