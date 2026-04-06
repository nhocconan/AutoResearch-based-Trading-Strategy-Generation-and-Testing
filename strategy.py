#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Spike + Range Filter
Hypothesis: Camarilla levels from daily pivot identify intraday support/resistance.
In ranging markets (low ADX), fade extremes at R3/S3 for mean reversion.
In trending markets (high ADX), breakout through R4/S4 continues trend.
Volume spike confirms institutional participation. Works in bull/bear via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # R2 = Close + 0.5 * (High - Low)
    # R1 = Close + 0.25 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 0.25 * (High - Low)
    # S2 = Close - 0.5 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            hl_range = high_1d[i] - low_1d[i]
            camarilla_r4[i] = close_1d[i] + 1.5 * hl_range
            camarilla_r3[i] = close_1d[i] + 1.0 * hl_range
            camarilla_s3[i] = close_1d[i] - 1.0 * hl_range
            camarilla_s4[i] = close_1d[i] - 1.5 * hl_range
    
    # Calculate ADX for regime detection
    # +DM, -DM, TR
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr_1d = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    if len(tr_1d) >= 14:
        atr_1d = wilders_smoothing(tr_1d, 14)
        plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
        minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = wilders_smoothing(dx_1d, 14)
    else:
        adx_1d = np.full(len(high_1d), np.nan)
    
    # Align 1d data to 6h timeframe
    camarilla_r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike detection (current volume > 2x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 50  # Need enough data for indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(camarilla_r3_6h[i]) or 
            np.isnan(camarilla_s3_6h[i]) or np.isnan(camarilla_r4_6h[i]) or
            np.isnan(camarilla_s4_6h[i]) or np.isnan(adx_1d_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: reverse signal or stoploss
            if (close[i] < camarilla_s3_6h[i] or  # Mean reversion exit
                close[i] > camarilla_r4_6h[i] or   # Breakout continuation exit
                close[i] < entry_price - 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: reverse signal or stoploss
            if (close[i] > camarilla_r3_6h[i] or   # Mean reversion exit
                close[i] < camarilla_s4_6h[i] or   # Breakout continuation exit
                close[i] > entry_price + 2.5 * atr[i]):  # Stoploss
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries - regime dependent
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Volume filter
                if not volume_spike[i]:
                    signals[i] = 0.0
                    bars_since_entry += 1
                    continue
                
                # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
                if adx_1d_6h[i] > 25:  # Trending regime - breakout
                    # Long: break above R4 with volume
                    if close[i] > camarilla_r4_6h[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                    # Short: break below S4 with volume
                    elif close[i] < camarilla_s4_6h[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.0
                        bars_since_entry += 1
                elif adx_1d_6h[i] < 20:  # Ranging regime - mean reversion
                    # Long: bounce from S3 with volume
                    if close[i] < camarilla_s3_6h[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                    # Short: bounce from R3 with volume
                    elif close[i] > camarilla_r3_6h[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.0
                        bars_since_entry += 1
                else:  # Transition regime - no trade
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals