#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 12h volume confirmation + 1d ADX trend filter
# Camarilla levels (R3/S3, R4/S4) provide institutional support/resistance
# Breakout above R4 or below S4 with volume confirms strong momentum
# 12h ADX > 25 filters for trending markets, avoids false breakouts in chop
# Works in bull/bear: ADX regime filter adapts, Camarilla breakouts capture strong moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_12h_1d_camarilla_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for ATR
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed DM
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_12h
    minus_di = 100 * minus_dm_smooth / atr_12h
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx_12h = wilders_smoothing(dx, 14)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels
    r3_1d = pivot_1d + range_1d * 1.1 / 4.0
    s3_1d = pivot_1d - range_1d * 1.1 / 4.0
    r4_1d = pivot_1d + range_1d * 1.1 / 2.0
    s4_1d = pivot_1d - range_1d * 1.1 / 2.0
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar close)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Align 1d Camarilla levels to 6h timeframe (wait for 1d bar close)
    # Need to shift by 1 to use previous day's levels
    r3_1d_shifted = np.concatenate([[np.nan], r3_1d[:-1]])
    s3_1d_shifted = np.concatenate([[np.nan], s3_1d[:-1]])
    r4_1d_shifted = np.concatenate([[np.nan], r4_1d[:-1]])
    s4_1d_shifted = np.concatenate([[np.nan], s4_1d[:-1]])
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d_shifted)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d_shifted)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d_shifted)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d_shifted)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(avg_volume_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 12h average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_12h_aligned[i] > 25.0
        
        if position == 1:  # Long position
            # Exit: price closes below R3 OR ADX drops below 20 (trend weakening)
            if close[i] < r3_1d_aligned[i] or adx_12h_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 OR ADX drops below 20 (trend weakening)
            if close[i] > s3_1d_aligned[i] or adx_12h_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation in trending market
            if trending and volume_confirmed:
                # Breakout above R4 = long
                if close[i] > r4_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below S4 = short
                elif close[i] < s4_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals