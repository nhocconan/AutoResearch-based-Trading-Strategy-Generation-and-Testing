# 1d_1w_1d_Pivot_R3S3_Reversal_V1
# Hypothesis: On 1d timeframe, trade reversals at weekly Camarilla S3/R3 levels with volume confirmation and ADX regime filter.
# In ranging markets (ADX < 25), price reverses at S3/R3; in trending markets (ADX > 25), price breaks through S4/R4.
# Targets 15-25 trades/year by requiring confluence of level, volume, and regime filter.
# Works in both bull and bear markets due to adaptive regime filtering.
# Uses weekly HTF for context (not for entry levels) to avoid overfitting.

name = "1d_1w_1d_Pivot_R3S3_Reversal_V1"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop for regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for entry levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S3, R3, S4, R4
    s3_1d = close_1d - (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 6)
    s4_1d = close_1d - (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align 1d levels to 1d timeframe (identity but for safety)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Calculate 1d ADX for trend/ranging filter (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate weekly ADX for trend context (optional filter)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    
    # Directional Movement for weekly
    up_move_w = high_1w[1:] - high_1w[:-1]
    down_move_w = low_1w[:-1] - low_1w[1:]
    plus_dm_w = np.where((up_move_w > down_move_w) & (up_move_w > 0), up_move_w, 0.0)
    minus_dm_w = np.where((down_move_w > up_move_w) & (down_move_w > 0), down_move_w, 0.0)
    plus_dm_w = np.concatenate([[np.nan], plus_dm_w])
    minus_dm_w = np.concatenate([[np.nan], minus_dm_w])
    
    # Smoothed TR and DM using Wilder smoothing for weekly
    atr_w = smooth_wilder(tr_w, 14)
    plus_di_w = 100 * smooth_wilder(plus_dm_w, 14) / atr_w
    minus_di_w = 100 * smooth_wilder(minus_dm_w, 14) / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = smooth_wilder(dx_w, 14)
    
    # Align weekly ADX to 1d timeframe
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(adx_w_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Ranging market (ADX < 25): reverse at S3/R3
            if adx_aligned[i] < 25:
                # Long near S3 with volume confirmation
                if (close[i] <= s3_aligned[i] * 1.005 and 
                    close[i] >= s3_aligned[i] * 0.995 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short near R3 with volume confirmation
                elif (close[i] >= r3_aligned[i] * 0.995 and 
                      close[i] <= r3_aligned[i] * 1.005 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
            # Trending market (ADX > 25): breakout at S4/R4
            elif adx_aligned[i] > 25:
                # Long breakout above R4 with volume
                if (close[i] > r4_aligned[i] * 1.005 and 
                    volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below S4 with volume
                elif (close[i] < s4_aligned[i] * 0.995 and 
                      volume[i] > 2.0 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] >= r3_aligned[i] * 0.995) or \
               (adx_aligned[i] > 25 and close[i] < s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reverse at opposite level or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] <= s3_aligned[i] * 1.005) or \
               (adx_aligned[i] > 25 and close[i] > r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals