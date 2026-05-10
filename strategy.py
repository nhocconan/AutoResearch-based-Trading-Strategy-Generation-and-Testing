#!/usr/bin/env python3
# 12h_WeeklyPivot_VolumeRegime_Signal
# Hypothesis: Uses weekly pivot points (R1/S1) from 1w data, combined with daily volume regime filter (volume > 1.5x 20-day average) and ADX trend filter on 1d.
# Designed for low-frequency, high-conviction trades on 12h timeframe to avoid fee drag.
# Works in bull markets via long breakouts above weekly R1 in uptrend, and in bear markets via short breakdowns below weekly S1 in downtrend.
# Volume regime avoids low-liquidity chop; ADX ensures trades align with stronger trends.
# Position size: 0.25 to balance risk and return. Target: 20-40 trades/year.

name = "12h_WeeklyPivot_VolumeRegime_Signal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for volume regime and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points: R1, S1 from prior week
    # Formula: R1 = (2 * PP) - Low, S1 = (2 * PP) - High, where PP = (High + Low + Close)/3
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    r1 = (2 * pp) - prev_weekly_low
    s1 = (2 * pp) - prev_weekly_high
    
    # Align weekly pivot levels to 12h timeframe (waits for weekly bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily volume regime: volume > 1.5x 20-day average
    vol_ma_20d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    volume_regime = volume > (vol_ma_20d_aligned * 1.5)
    
    # Daily ADX trend filter (14-period)
    # TR = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close_1d = df_1d['close'].shift(1).values
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close_1d)
    tr3 = np.abs(df_1d['low'] - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM = high - prev_high (if > prev_low - low and > 0)
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    plus_dm = df_1d['high'] - prev_high_1d
    minus_dm = prev_low_1d - df_1d['low']
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI = 100 * (+DM_smoothed / TR_smoothed), -DI = 100 * (-DM_smoothed / TR_smoothed)
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX = 100 * |(+DI - -DI)| / ((+DI) + (-DI))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    # ADX = smoothed DX
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume and ADX arrays aligned to 12h
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Warmup for vol MA, ADX
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_regime_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade if ADX > 20 (trending market) and volume regime is active
        strong_trend = adx_aligned[i] > 20
        vol_ok = volume_regime_aligned[i] > 0.5  # boolean as float
        
        if position == 0:
            # Long: price breaks above weekly R1, in uptrend, with volume
            if close[i] > r1_aligned[i] and strong_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, in downtrend, with volume
            elif close[i] < s1_aligned[i] and strong_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below weekly R1 or trend weakens
            if close[i] < r1_aligned[i] or not strong_trend or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above weekly S1 or trend weakens
            if close[i] > s1_aligned[i] or not strong_trend or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals