# 12h_WILLIAMS_ALLIGATOR_Trend_With_1w_Filter
# Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMMA) on 12h for trend direction.
# Filtered by 1w EMA200 trend: long only when price > 1w EMA200, short only when price < 1w EMA200.
# Alligator lines must be aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend).
# ADX > 20 confirms trend strength to avoid whipsaw.
# Uses SMMA (Smoothed Moving Average) for Alligator lines.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WILLIAMS_ALLIGATOR_Trend_With_1w_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    
    # Initial value: simple average of first 'period' values
    result[period-1] = np.mean(arr[:period])
    
    # Wilder's smoothing: SMMA(t) = (SMMA(t-1) * (period-1) + price(t)) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Williams Alligator on 12h data
    # Jaw: 13-period SMMA of median price, shifted 8 bars forward
    # Teeth: 8-period SMMA of median price, shifted 5 bars forward  
    # Lips: 5-period SMMA of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Shift forward as per Williams Alligator definition
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    # Jaw shifted 8 bars forward
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    # Teeth shifted 5 bars forward
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    # Lips shifted 3 bars forward
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # ADX for trend strength confirmation
    period = 14
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values using Wilder's smoothing
    atr = np.full_like(high, np.nan)
    dm_plus_smooth = np.full_like(high, np.nan)
    dm_minus_smooth = np.full_like(high, np.nan)
    
    # Initial values
    if len(high) >= period:
        atr[period] = np.nanmean(tr[1:period+1])
        dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period + 1, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.full_like(high, np.nan)
    di_minus = np.full_like(high, np.nan)
    dx = np.full_like(high, np.nan)
    
    valid = ~np.isnan(atr) & (atr != 0)
    di_plus[valid] = (dm_plus_smooth[valid] / atr[valid]) * 100
    di_minus[valid] = (dm_minus_smooth[valid] / atr[valid]) * 100
    
    dx_valid = valid & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])) * 100
    
    # ADX (smoothed DX)
    adx = np.full_like(high, np.nan)
    if len(high) >= 2 * period:
        adx[2*period] = np.nanmean(dx[period+1:2*period+1])
        for i in range(2*period + 1, len(high)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2*period + 1, 50, 13)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment checks
        jaw_gt_teeth = jaw[i] > teeth[i]
        teeth_gt_lips = teeth[i] > lips[i]
        jaw_lt_teeth = jaw[i] < teeth[i]
        teeth_lt_lips = teeth[i] < lips[i]
        
        # Trend from 1w EMA200
        uptrend_filter = close[i] > ema200_1w_aligned[i]
        downtrend_filter = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: uptrend filter + Alligator aligned up + ADX > 20
            if uptrend_filter and jaw_gt_teeth and teeth_gt_lips and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short: downtrend filter + Alligator aligned down + ADX > 20
            elif downtrend_filter and jaw_lt_teeth and teeth_lt_lips and adx[i] > 20:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator alignment breaks or trend filter fails
            if not (jaw_gt_teeth and teeth_gt_lips) or not uptrend_filter or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator alignment breaks or trend filter fails
            if not (jaw_lt_teeth and teeth_lt_lips) or not downtrend_filter or adx[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals