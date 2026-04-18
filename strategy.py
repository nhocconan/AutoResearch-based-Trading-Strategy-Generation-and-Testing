#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + Daily ATR + Volume Filter
# Uses weekly pivot points for structural support/resistance and daily ATR for volatility filtering.
# Long when price breaks above weekly R1 with volume and volatility confirmation.
# Short when price breaks below weekly S1 with volume and volatility confirmation.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (breakouts above weekly R1) and bear markets (breakouts below weekly S1).
name = "6h_WeeklyPivot_DailyATR_Volume_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)
    r3 = high_w + 2 * (pivot - low_w)
    s3 = low_w - 2 * (high_w - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for ATR filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14-period)
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing (EMA with alpha=1/14)
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold = atr * atr_mult
    
    # Align daily ATR threshold to 6h timeframe
    atr_threshold_6h = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(atr_threshold_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: current ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_6h[i]) and atr_threshold_6h[i] > 0
        
        if position == 0:
            # Long: price breaks above weekly R1 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r1_6h[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < s1_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below weekly S1 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] < s1_6h[i] or (np.isnan(atr_threshold_6h[i]) or atr_threshold_6h[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above weekly R1 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] > r1_6h[i] or (np.isnan(atr_threshold_6h[i]) or atr_threshold_6h[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals