#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and daily ADX trend filter.
# Trades on breakouts of key Camarilla levels (H3/L3) with volume spike, filtered by daily trend direction.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Uses daily structure for trend and levels,
# 12h volume surge for momentum confirmation. Works in bull/bear by following higher timeframe trends.

name = "12h_Camarilla_H3L3_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (based on previous day)
    # H3 = close + 1.1*(high - low)
    # L3 = close - 1.1*(high - low)
    # Using previous day's values to avoid look-ahead
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan  # First value invalid
    camarilla_l3[0] = np.nan
    
    # Calculate daily ADX (14-period) for trend strength
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]  # First value
    tr = true_range(high_1d, low_1d, close_prev_1d)
    
    # Directional movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM
    def rma(arr, period):
        """Wilder's smoothing (RMA)"""
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = rma(tr, 14)
    plus_di = 100 * rma(plus_dm, 14) / atr
    minus_di = 100 * rma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = rma(dx, 14)
    
    # Align daily indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 12h data for volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Volume spike: 1.5x 20-period EMA
    vol_ema = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_12h > (vol_ema * 1.5)
    
    # Align volume spike to 12h timeframe
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 for trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 0 and is_trending:
            # Enter long: price breaks above H3 + volume spike
            if close[i] > camarilla_h3_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below L3 + volume spike
            elif close[i] < camarilla_l3_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below L3 or ADX weakens
            if close[i] < camarilla_l3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above H3 or ADX weakens
            if close[i] > camarilla_h3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals