#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v2
# Strategy: 12h Camarilla pivot breakout with daily volume confirmation and ADX trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily chart act as strong support/resistance.
# Breakouts above/below these levels with volume confirmation and trend alignment (ADX > 25)
# capture sustained moves. Works in both bull (breakouts continue) and bear (fades at pivots).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L4 = C + (H - L) * 1.1 / 2
    # H4 = C + (H - L) * 1.1 / 2
    # L3 = C + (H - L) * 1.1 / 4
    # H3 = C + (H - L) * 1.1 / 4
    # L2 = C + (H - L) * 1.1 / 6
    # H2 = C + (H - L) * 1.1 / 6
    # L1 = C + (H - L) * 1.1 / 12
    # H1 = C + (H - L) * 1.1 / 12
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h4_1d = close_1d + range_1d * 1.1 / 2
    l4_1d = close_1d - range_1d * 1.1 / 2
    h3_1d = close_1d + range_1d * 1.1 / 4
    l3_1d = close_1d - range_1d * 1.1 / 4
    h2_1d = close_1d + range_1d * 1.1 / 6
    l2_1d = close_1d - range_1d * 1.1 / 6
    h1_1d = close_1d + range_1d * 1.1 / 12
    l1_1d = close_1d - range_1d * 1.1 / 12
    
    # Align to 12h timeframe (use previous day's levels)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h2_1d_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    l2_1d_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    h1_1d_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    l1_1d_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    
    # Daily ADX for trend filter (14-period)
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    di_plus_1d = wilders_smoothing(dm_plus, 14)
    di_minus_1d = wilders_smoothing(dm_minus, 14)
    
    # DI values to avoid division by zero
    di_sum = di_plus_1d + di_minus_1d
    dx_1d = np.where(di_sum != 0, 100 * np.abs(di_plus_1d - di_minus_1d) / di_sum, 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = vol_1d > (1.5 * vol_avg_20)
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > h4_1d_aligned[i]
        breakout_down = close[i] < l4_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_ok = vol_confirm_1d_aligned[i] if not np.isnan(vol_confirm_1d_aligned[i]) else False
        
        # Entry logic: Breakout + volume + strong trend
        if breakout_up and vol_ok and strong_trend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_ok and strong_trend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or loss of trend/volume
        elif position == 1 and (breakout_down or not vol_ok or adx_1d_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up or not vol_ok or adx_1d_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals