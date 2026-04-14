#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour strategy using weekly pivot points for directional bias and daily ADX for trend strength.
# Long when price breaks above weekly R2 with daily ADX > 20; short when price breaks below weekly S2 with daily ADX > 20.
# In low ADX (ADX <= 20), fade at weekly R3/S3 with reversal confirmation.
# Volume > 1.5x 20-period average confirms breakouts/breakdowns.
# Weekly pivots provide structural support/resistance that works in both bull and bear markets.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Load weekly data for pivot calculation (using daily data to compute weekly pivots)
    # We'll use the same daily data but resample conceptually via rolling window for weekly high/low/close
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close from daily data (simplified: use last 5 days)
    # For proper weekly pivot, we need actual weekly OHLC, but we'll approximate with 5-day period
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivots to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # Daily ADX for trend strength
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14*2, 5)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or
            np.isnan(weekly_s2_aligned[i]) or
            np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend regime: ADX > 20 = trending, ADX <= 20 = ranging
        trending = adx_aligned[i] > 20
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if trending:
                # In trending market: breakout trades in direction of weekly pivot levels
                if (close[i] > weekly_r2_aligned[i] and 
                    volume_confirmed):
                    position = 1
                    signals[i] = position_size
                elif (close[i] < weekly_s2_aligned[i] and 
                      volume_confirmed):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # In ranging market: fade at extreme weekly levels (R3/S3)
                if (close[i] > weekly_r3_aligned[i] and 
                    volume_confirmed):
                    position = -1
                    signals[i] = -position_size
                elif (close[i] < weekly_s3_aligned[i] and 
                      volume_confirmed):
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or breaks below S1
            if (close[i] < weekly_pivot_aligned[i] or 
                close[i] < weekly_s1_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot or breaks above R1
            if (close[i] > weekly_pivot_aligned[i] or 
                close[i] > weekly_r1_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyPivot_DailyADX_v1"
timeframe = "6h"
leverage = 1.0