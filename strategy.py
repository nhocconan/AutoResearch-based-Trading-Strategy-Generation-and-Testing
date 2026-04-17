#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe strategy using weekly Camarilla pivot levels (R4/S4) for breakout direction
combined with 1d ADX trend filter and 6h volume confirmation. 
Weekly Camarilla R4/S4 represent strong weekly support/resistance - breaks indicate institutional interest.
1d ADX > 25 ensures we only trade in trending markets, reducing whipsaw.
Volume confirmation ensures breakouts have participation.
Designed to work in both bull (breakouts above R4) and bear (breakdowns below S4) markets.
Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # need enough for weekly lookback
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly Camarilla pivots (based on prior week OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4_1w = close_1w + 1.5 * (high_1w - low_1w)
    camarilla_s4_1w = close_1w - 1.5 * (high_1w - low_1w)
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high_vals, low_vals, close_vals, window=14):
        # True Range
        tr1 = np.abs(high_vals[1:] - low_vals[1:])
        tr2 = np.abs(high_vals[1:] - close_vals[:-1])
        tr3 = np.abs(low_vals[1:] - close_vals[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        up_move = high_vals[1:] - high_vals[:-1]
        down_move = low_vals[:-1] - low_vals[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/window)
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder smoothing
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = WilderSmooth(tr, window)
        plus_dm_smooth = WilderSmooth(plus_dm, window)
        minus_dm_smooth = WilderSmooth(minus_dm, window)
        
        # Directional Indicators
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderSmooth(dx, window)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 6h volume 20-period average
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (6h)
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_6h)  # volume already 6h, align via 1d index
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for weekly and ADX calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r4_1w_aligned[i]) or 
            np.isnan(camarilla_s4_1w_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_6h_aligned[i]
        
        # ADX filter: only trade when trending (ADX > 25)
        trending_market = adx_14_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with volume and trend
            if (close[i] > camarilla_r4_1w_aligned[i] and 
                volume_confirmed and 
                trending_market):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with volume and trend
            elif (close[i] < camarilla_s4_1w_aligned[i] and 
                  volume_confirmed and 
                  trending_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Camarilla R3/R4 midpoint
            camarilla_r3_1w = camarilla_r4_1w_aligned[i] - (camarilla_r4_1w_aligned[i] - camarilla_s4_1w_aligned[i]) * 0.125
            if close[i] < camarilla_r3_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Camarilla S3/S4 midpoint
            camarilla_s3_1w = camarilla_s4_1w_aligned[i] + (camarilla_r4_1w_aligned[i] - camarilla_s4_1w_aligned[i]) * 0.125
            if close[i] > camarilla_s3_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wCamarilla_R4S4_Breakout_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0