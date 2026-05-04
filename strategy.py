#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX(14) > 25 trend filter and volume spike (>1.5x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure (breakout = new 6h close above R3 or below S3)
# 1d ADX(14) > 25 ensures we only trade in strong trending markets (avoids ranging/whipsaw)
# Volume confirmation ensures breakout has sufficient participation (>1.5x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 60-120 total trades over 4 years = 15-30/year for 6h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias)

name = "6h_Camarilla_R3S3_1dADX_VolumeSpike"
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
    
    # Get 1d data for ADX trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) trend filter from prior completed 1d bar
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate True Range (TR)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement (DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM
    tr_smooth = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Shift ADX by 1 to use only prior completed 1d bar (no look-ahead)
    adx_shifted = np.roll(adx, 1)
    adx_shifted[0] = np.nan
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L) * 1.1/4, S3 = C - (H-L) * 1.1/4
    # R4 = C + (H-L) * 1.1/2, S4 = C - (H-L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    camarilla_r4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_s4 = close_1d - range_1d * 1.1 / 2.0
    
    # Shift Camarilla levels by 1 to use only prior completed 1d bar (no look-ahead)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r4_shifted = np.roll(camarilla_r4, 1)
    camarilla_s4_shifted = np.roll(camarilla_s4, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    camarilla_r4_shifted[0] = np.nan
    camarilla_s4_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + ADX > 25 + volume spike
            if close[i] > camarilla_r3_aligned[i] and adx_aligned[i] > 25.0 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + ADX > 25 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and adx_aligned[i] > 25.0 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR ADX drops below 20 (trend weakening)
            if close[i] < camarilla_s3_aligned[i] or adx_aligned[i] < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR ADX drops below 20 (trend weakening)
            if close[i] > camarilla_r3_aligned[i] or adx_aligned[i] < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals