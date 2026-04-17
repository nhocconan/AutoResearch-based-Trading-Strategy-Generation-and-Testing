#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with volume spike and ADX trend filter.
Long when price breaks above R3 (1d) AND 4h volume > 2.0x 20-bar average AND ADX > 25 (trending).
Short when price breaks below S3 (1d) AND 4h volume > 2.0x 20-bar average AND ADX > 25 (trending).
Exit when price touches 1d pivot point (PP) or opposite Camarilla level (S3 for long, R3 for short).
Uses 1d for Camarilla levels/PP and ADX regime, 4h for execution and volume confirmation.
Designed to capture strong breakouts in trending markets with volume confirmation. Target: 15-30 trades/year per symbol.
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
    
    # Get 1d data for Camarilla levels and ADX regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (wider bands for stronger breakouts)
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    # PP = (High + Low + Close)/3
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng / 4
    s3 = close_1d - 1.1 * rng / 4
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1d ADX (trend strength filter)
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # +DI = 100 * EWMA(+DM) / ATR
    # -DI = 100 * EWMA(-DM) / ATR
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX)
    
    # True Range
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial TR sum
    tr_sum = np.zeros_like(tr1)
    tr_sum[period-1] = np.nansum(tr1[:period])
    
    # Initial DM sums
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    plus_dm_sum[period-1] = np.nansum(plus_dm[:period])
    minus_dm_sum[period-1] = np.nansum(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr1)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr1[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm[i]
    
    # Avoid division by zero
    tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
    plus_di = 100 * plus_dm_sum / tr_sum_safe
    minus_di = 100 * minus_dm_sum / tr_sum_safe
    
    # DX and ADX
    di_sum = plus_di + minus_di
    di_sum_safe = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum_safe
    
    # ADX: EMA of DX
    adx = np.zeros_like(dx)
    adx[period-1] = np.nanmean(dx[:period]) if period <= len(dx) else 0
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 20-bar average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: ADX > 25 indicates trending market (breakout favorable)
        trending_market = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Exit conditions: touch pivot or opposite level
        touch_pp = abs(close[i] - pp_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < s3_aligned[i]) or \
                         (position == -1 and close[i] > r3_aligned[i])
        
        if position == 0:
            # Long: break above R3 with volume confirmation and trending market
            if (breakout_r3 and volume_confirmed and trending_market):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume confirmation and trending market
            elif (breakout_s3 and volume_confirmed and trending_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch pivot or break below S3
            if (touch_pp or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch pivot or break above R3
            if (touch_pp or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0