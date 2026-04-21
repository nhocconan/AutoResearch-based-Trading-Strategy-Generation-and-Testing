#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_v1
Hypothesis: On 1d timeframe, price breaking above Camarilla R1 or below S1 from previous day indicates intraday momentum, with weekly EMA50 filter for trend alignment and ADX(14) < 20 for ranging regime avoidance. Uses discrete sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels, 1w for EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar: use same day's data (will be filtered by min_periods anyway)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 12
    camarilla_s1 = prev_close - range_1d * 1.1 / 12
    camarilla_r3 = prev_close + range_1d * 1.1 / 4
    camarilla_s3 = prev_close - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1w EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d ADX(14) for regime filter (avoid ranging markets) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_50 = ema_50_aligned[i]
        adx_val = adx[i]
        
        # Regime filter: avoid ranging markets (ADX < 20 = ranging)
        is_trending_regime = adx_val >= 20
        
        if position == 0:
            # Long: price breaks above R1 + above EMA50 + trending regime
            if price > r1 and price > ema_50 and is_trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below EMA50 + trending regime
            elif price < s1 and price < ema_50 and is_trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters R1-S1 range or trend weakens
            if position == 1:
                if price < r1 or adx_val < 15:  # Re-enter range or weak trend
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > s1 or adx_val < 15:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0