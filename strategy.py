#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ADX trend strength + 1d Camarilla pivot breakouts with volume confirmation
# Long when price breaks above R3 (1d) AND 12h ADX > 25 (strong trend) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below S3 (1d) AND 12h ADX > 25 AND volume > 1.5 * avg_volume(20)
# Exit when price retouches the pivot point (PP) OR ADX weakens (< 20) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# ADX filters for genuine trend strength to avoid false breakouts in ranging markets
# Camarilla R3/S3 provides precise breakout levels from 1d structure
# Volume confirmation ensures breakout legitimacy
# Works in bull markets (buying R3 breakouts in uptrend) and bear markets (selling S3 breakouts in downtrend)

name = "6h_ADX25_CamarillaR3S3_1dPP_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar for pivots
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = PP + (High - Low) * 1.1/12
    # R2 = PP + (High - Low) * 1.1/6
    # R3 = PP + (High - Low) * 1.1/4
    # S1 = PP - (High - Low) * 1.1/12
    # S2 = PP - (High - Low) * 1.1/6
    # S3 = PP - (High - Low) * 1.1/4
    rng = high_1d - low_1d
    r3 = pp + rng * 1.1 / 4.0
    s3 = pp - rng * 1.1 / 4.0
    pp_val = pp  # for exit condition
    
    # Align Camarilla levels to 6h timeframe (wait for completed daily bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_val)
    
    # Get 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:  # Need enough for ADX calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h timeframe
    # True Range (TR) = max[(H-L), abs(H-PC), abs(L-PC)]
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just H-L
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 6h timeframe (wait for completed 12h bar)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND ADX > 25 (strong trend) AND volume confirmation, in session
            if (close[i] > r3_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND ADX > 25 (strong trend) AND volume confirmation, in session
            elif (close[i] < s3_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches pivot point OR ADX weakens (< 20) OR volume drops below average
            if (close[i] <= pp_aligned[i] or adx_aligned[i] < 20 or volume[i] < avg_volume_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches pivot point OR ADX weakens (< 20) OR volume drops below average
            if (close[i] >= pp_aligned[i] or adx_aligned[i] < 20 or volume[i] < avg_volume_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals