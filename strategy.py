#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 reversal with 1d ADX trend filter and volume confirmation.
Short when price rejects at R3 (overbought) in strong uptrend (ADX>25) with volume confirmation.
Long when price finds support at S3 (oversold) in strong downtrend (ADX>25) with volume confirmation.
Exit when price moves toward the mean (Camarilla pivot) or ADX weakens (<20).
Uses 1d HTF for ADX trend strength to avoid choppy markets, Camarilla R3/S3 for reversal zones,
volume to confirm institutional interest at extremes. Works in ranging markets (reversion) and
strong trends (fade the extreme, not the trend). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels (based on previous 6h bar)
    camarilla_pivot = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to avoid look-ahead
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        range_ = phigh - plow
        camarilla_pivot[i] = pivot
        camarilla_r3[i] = pivot + range_ * 1.1 / 4.0  # R3 level
        camarilla_s3[i] = pivot - range_ * 1.1 / 4.0  # S3 level
    
    # Calculate 1d ADX for trend strength filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 30, 20)  # Camarilla (needs 1), ADX, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pivot[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pivot = camarilla_pivot[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 1.3x 20-period MA
        vol_filter = volume[i] > 1.3 * vol_ma_val
        
        # Trend strength filters
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        
        if position == 0:
            # Long: Price at S3 support in strong downtrend + volume
            if price <= s3 and strong_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price at R3 resistance in strong uptrend + volume
            elif price >= r3 and strong_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price moves toward pivot (mean reversion) OR trend weakens
                if price >= pivot or weak_trend:
                    exit_signal = True
            elif position == -1:
                # Short exit: price moves toward pivot (mean reversion) OR trend weakens
                if price <= pivot or weak_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Reversal_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0