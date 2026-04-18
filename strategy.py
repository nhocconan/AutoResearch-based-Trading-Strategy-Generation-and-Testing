#!/usr/bin/env python3
"""
6h Bollinger Band Reversal with 12h ADX Trend Filter
Trades reversals from Bollinger Bands (20,2) when 12h ADX > 25 indicates trending market.
In strong trends, prices often revert to the mean after touching Bollinger Bands.
Uses Bollinger Band width to filter out choppy markets.
Designed for low trade frequency with edge in both bull and bear markets by fading extremes in trending conditions.
"""

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
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Bollinger Band Width for regime filter
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14) on 12h
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def smoothed_ma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(arr[1:period])
            # Subsequent values are Wilder's smoothing
            for i in range(period, len(arr)):
                if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smoothed_ma(tr, 14)
    dm_plus_smooth = smoothed_ma(dm_plus, 14)
    dm_minus_smooth = smoothed_ma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_ma(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume filter (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        
        # Only trade when ADX indicates trending market (>25) and not in extreme chop
        if adx_val > 25 and bb_width[i] < bb_width_ma[i] * 1.5:
            if position == 0:
                # Long: price touches or goes below lower BB with volume
                if price <= bb_lower[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price touches or goes above upper BB with volume
                elif price >= bb_upper[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Long position: exit when price returns to middle BB
                signals[i] = 0.25
                if price >= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
            
            elif position == -1:
                # Short position: exit when price returns to middle BB
                signals[i] = -0.25
                if price <= bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
        else:
            # No position or exit if conditions not met
            if position != 0:
                # Exit if trend weakens or volatility expands too much
                if adx_val < 20 or bb_width[i] > bb_width_ma[i] * 2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Bollinger_Reversal_12hADX_TrendFilter"
timeframe = "6h"
leverage = 1.0