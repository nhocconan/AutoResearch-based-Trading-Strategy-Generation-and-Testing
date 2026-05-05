#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume spike and 1w ADX trend filter
# Long when price > Alligator Jaw AND volume > 2.0x 20-period average AND 1w ADX > 25 (trending)
# Short when price < Alligator Jaw AND volume > 2.0x 20-period average AND 1w ADX > 25 (trending)
# Exit when price crosses Alligator Teeth OR 1w ADX < 20 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Williams Alligator identifies trend via smoothed medians (Jaw/Teeth/Lips). Volume confirms institutional participation.
# 1w ADX filter ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
# Works in bull markets via longs and bear markets via shorts by following the primary trend.

name = "12h_Williams_Alligator_1wADX25_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need enough for smoothed medians
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaw (Blue): 13-period SMMA of median price, shifted 8 bars
    # Teeth (Red): 8-period SMMA of median price, shifted 5 bars
    # Lips (Green): 5-period SMMA of median price, shifted 3 bars
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w data
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        if len(high) < period + 1:
            return np.full_like(close, np.nan, dtype=float)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed TR, DM+
        def smma_series(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan, dtype=float)
            result = np.full_like(values, np.nan, dtype=float)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        atr = smma_series(tr, period)
        dm_plus_smooth = smma_series(dm_plus, period)
        dm_minus_smooth = smma_series(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = smma_series(dx, period)
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    strong_trend_1w = adx_1w > 25
    weak_trend_1w = adx_1w < 20  # For exit
    
    # Align 1w ADX to 12h timeframe
    strong_trend_1w_aligned = align_htf_to_ltf(prices, df_1w, strong_trend_1w.astype(float))
    weak_trend_1w_aligned = align_htf_to_ltf(prices, df_1w, weak_trend_1w.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(strong_trend_1w_aligned[i]) or 
            np.isnan(weak_trend_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Jaw AND volume spike AND strong 1w uptrend (ADX>25)
            if (close[i] > jaw_aligned[i] and 
                volume_filter[i] and 
                strong_trend_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Jaw AND volume spike AND strong 1w downtrend (ADX>25)
            elif (close[i] < jaw_aligned[i] and 
                  volume_filter[i] and 
                  strong_trend_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Teeth OR 1w trend weakens (ADX<20)
            if (close[i] < teeth_aligned[i] or 
                weak_trend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Teeth OR 1w trend weakens (ADX<20)
            if (close[i] > teeth_aligned[i] or 
                weak_trend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals