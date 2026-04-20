#!/usr/bin/env python3
# 12h_1w_1d_Pivot_R3S3_Fade_Reverse_V1
# Hypothesis: On 12h timeframe, trade reversals at 1d and 1w Camarilla R3/S3 levels with volume confirmation.
# Uses 1d ADX to filter ranging (ADX < 20) for reversals at 1d R3/S3, and 1w ADX > 25 for breakouts at 1w R4/S4.
# Aims for 12-30 trades/year by requiring confluence of level, volume, and regime filter.
# Designed to work in both bull and bear markets by adapting to volatility regimes.

name = "12h_1w_1d_Pivot_R3S3_Fade_Reverse_V1"
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
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3/S3 for reversals in ranging markets)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    r3_1d = close_1d + (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 6)
    r4_1d = close_1d + (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 4)
    
    # Calculate 1w Camarilla pivot levels (R4/S4 for breakouts in trending markets)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w = typical_price_1w
    range_1w = high_1w - low_1w
    
    r4_1w = close_1w + (range_1w * 1.1 / 4)
    s4_1w = close_1w - (range_1w * 1.1 / 4)
    
    # Align 1d levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Align 1w levels to 12h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate 1d ADX for ranging filter (ADX < 20)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1d = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr_1d
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = smooth_wilder(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1w ADX for trending filter (ADX > 25)
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    
    up_move_w = high_1w[1:] - high_1w[:-1]
    down_move_w = low_1w[:-1] - low_1w[1:]
    plus_dm_w = np.where((up_move_w > down_move_w) & (up_move_w > 0), up_move_w, 0.0)
    minus_dm_w = np.where((down_move_w > up_move_w) & (down_move_w > 0), down_move_w, 0.0)
    plus_dm_w = np.concatenate([[np.nan], plus_dm_w])
    minus_dm_w = np.concatenate([[np.nan], minus_dm_w])
    
    atr_1w = smooth_wilder(tr_w, 14)
    plus_di_w = 100 * smooth_wilder(plus_dm_w, 14) / atr_1w
    minus_di_w = 100 * smooth_wilder(minus_dm_w, 14) / atr_1w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_1w = smooth_wilder(dx_w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Ranging market (1d ADX < 20): fade at 1d R3/S3
            if adx_1d_aligned[i] < 20:
                # Long near S3 with volume confirmation
                if (close[i] <= s3_1d_aligned[i] * 1.005 and 
                    close[i] >= s3_1d_aligned[i] * 0.995 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short near R3 with volume confirmation
                elif (close[i] >= r3_1d_aligned[i] * 0.995 and 
                      close[i] <= r3_1d_aligned[i] * 1.005 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            # Trending market (1w ADX > 25): breakout at 1w R4/S4
            elif adx_1w_aligned[i] > 25:
                # Long breakout above R4 with volume
                if (close[i] > r4_1w_aligned[i] * 1.005 and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S4 with volume
                elif (close[i] < s4_1w_aligned[i] * 0.995 and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite level or ADX shifts to ranging
            if (adx_1d_aligned[i] < 20 and close[i] >= r3_1d_aligned[i] * 0.995) or \
               (adx_1w_aligned[i] > 25 and close[i] < s4_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse at opposite level or ADX shifts to ranging
            if (adx_1d_aligned[i] < 20 and close[i] <= s3_1d_aligned[i] * 1.005) or \
               (adx_1w_aligned[i] > 25 and close[i] > r4_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals