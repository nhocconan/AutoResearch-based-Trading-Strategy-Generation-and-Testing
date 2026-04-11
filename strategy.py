#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v2
# Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and 1d ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from daily timeframe act as strong support/resistance.
# Breakouts above/below these levels with volume > 1.5x 20-period average confirm institutional interest.
# ADX > 25 on daily timeframe filters for trending markets, avoiding choppy conditions.
# Designed for low trade frequency (~20-40/year) to minimize fee drag. Works in bull markets via long breakouts
# and bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    L4 = close_val + (range_val * 1.1 / 2)
    L3 = close_val + (range_val * 1.1 / 4)
    L2 = close_val + (range_val * 1.1 / 6)
    L1 = close_val + (range_val * 1.1 / 12)
    S1 = close_val - (range_val * 1.1 / 12)
    S2 = close_val - (range_val * 1.1 / 6)
    S3 = close_val - (range_val * 1.1 / 4)
    S4 = close_val - (range_val * 1.1 / 2)
    return L4, L3, L2, L1, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    L4_1d = np.full_like(close_1d, np.nan)
    L3_1d = np.full_like(close_1d, np.nan)
    L2_1d = np.full_like(close_1d, np.nan)
    L1_1d = np.full_like(close_1d, np.nan)
    S1_1d = np.full_like(close_1d, np.nan)
    S2_1d = np.full_like(close_1d, np.nan)
    S3_1d = np.full_like(close_1d, np.nan)
    S4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        L4, L3, L2, L1, S1, S2, S3, S4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        L4_1d[i] = L4
        L3_1d[i] = L3
        L2_1d[i] = L2
        L1_1d[i] = L1
        S1_1d[i] = S1
        S2_1d[i] = S2
        S3_1d[i] = S3
        S4_1d[i] = S4
    
    # Align Camarilla levels to 4h timeframe
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    L2_1d_aligned = align_htf_to_ltf(prices, df_1d, L2_1d)
    L1_1d_aligned = align_htf_to_ltf(prices, df_1d, L1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # 1d ADX for trend filter (requires high, low, close)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period:
            return np.full_like(high, np.nan)
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period-1] = np.nanmean(tr[0:period])
        plus_dm_sum = np.nansum(plus_dm[0:period])
        minus_dm_sum = np.nansum(minus_dm[0:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * (plus_dm_sum / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm_sum / atr[i]) if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Calculate ADX as smoothed DX
        adx = np.zeros_like(high)
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1]) if len(dx) >= 2*period-1 else np.nan
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(L4_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or np.isnan(L2_1d_aligned[i]) or 
            np.isnan(L1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or np.isnan(S2_1d_aligned[i]) or
            np.isnan(S3_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > L4_1d_aligned[i-1]  # Break above resistance level 4
        breakdown_down = close[i] < S4_1d_aligned[i-1]  # Break below support level 4
        
        # Entry conditions
        # Long: Breakout above L4 AND trending market AND volume confirmation
        if breakout_up and trending and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below S4 AND trending market AND volume confirmation
        elif breakdown_down and trending and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Camarilla level touch (touch L1 for short, S1 for long)
        elif position == 1 and close[i] <= S1_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= L1_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals