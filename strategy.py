#!/usr/bin/env python3
"""
4h Candlestick Engulfing Pattern with Volume and ADX Trend Filter.
Long when bullish engulfing + ADX > 25 (strong trend) + volume > 1.5x average.
Short when bearish engulfing + ADX > 25 + volume > 1.5x average.
Exit when opposite engulfing pattern forms or ADX drops below 20.
Designed to generate 20-50 trades/year per symbol with high-probability entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
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
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilders_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(arr)):
            result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
        return result
    
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period)
    
    return adx

def engulfing_pattern(open_price, high, low, close):
    """Detect bullish and bearish engulfing patterns"""
    n = len(close)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulfing: current green candle engulfs previous red candle
        if (close[i] > open_price[i] and  # Current candle bullish
            open_price[i-1] > close[i-1] and  # Previous candle bearish
            open_price[i] <= close[i-1] and  # Current open <= previous close
            close[i] >= open_price[i-1]):  # Current close >= previous open
            bullish[i] = True
            
        # Bearish engulfing: current red candle engulfs previous green candle
        elif (close[i] < open_price[i] and  # Current candle bearish
              open_price[i-1] < close[i-1] and  # Previous candle bullish
              open_price[i] >= close[i-1] and  # Current open >= previous close
              close[i] <= open_price[i-1]):  # Current close <= previous open
            bearish[i] = True
    
    return bullish, bearish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Engulfing patterns on 4h timeframe
    bullish_engulf, bearish_engulf = engulfing_pattern(open_price, high, low, close)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ADX (14+14) + volume MA (20) + engulfing (1)
    start_idx = max(28, 19, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        adx_val = adx_1d_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        is_bullish_engulf = bullish_engulf[i]
        is_bearish_engulf = bearish_engulf[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # ADX trend filter: >25 for strong trend, <20 for weak trend (exit)
        strong_trend = adx_val > 25
        weak_trend = adx_val < 20
        
        if position == 0:
            # Look for new entries only in strong trend with volume
            if strong_trend and vol_filter:
                if is_bullish_engulf:
                    signals[i] = size
                    position = 1
                elif is_bearish_engulf:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish engulfing OR weak trend
            if is_bearish_engulf or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish engulfing OR weak trend
            if is_bullish_engulf or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Engulfing_ADX_Volume_Filter"
timeframe = "4h"
leverage = 1.0