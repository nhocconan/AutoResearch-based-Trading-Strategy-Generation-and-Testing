# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 14-period RSI with 1d ADX trend filter and volume confirmation.
# RSI(14) < 30 for long entry, > 70 for short entry. ADX(14) > 25 on daily timeframe confirms trend strength.
# Volume > 1.5x average volume (20-period) confirms momentum. Exit on RSI crossing 50.
# Designed to work in both bull (trend following) and bear (mean reversion in range) markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def WilderSmoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(arr)):
                result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
            return result
        
        tr_smooth = WilderSmoothing(tr, period)
        dm_plus_smooth = WilderSmoothing(dm_plus, period)
        dm_minus_smooth = WilderSmoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = WilderSmoothing(dx, period)
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 14-period RSI on 6h timeframe
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)  # Insert 0 at beginning for same length
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for RSI
        def WilderSmoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            result[period-1] = np.mean(arr[:period])
            alpha = 1.0 / period
            for i in range(period, len(arr)):
                result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
            return result
        
        avg_gain = WilderSmoothing(gain, period)
        avg_loss = WilderSmoothing(loss, period)
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Average volume (20-period = 20*6h = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_1d_aligned[i]
        rsi_val = rsi[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: RSI < 30 (oversold) + trend filter + volume confirmation
            if (rsi_val < 30 and 
                trend_filter and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + trend filter + volume confirmation
            elif (rsi_val > 70 and 
                  trend_filter and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 (momentum fading)
            if rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50 (momentum fading)
            if rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_RSI_ADX_Volume"
timeframe = "6h"
leverage = 1.0