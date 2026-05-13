#!/usr/bin/env python3
# Hypothesis: 1h price action with 4h/1d regime filter - Use 4h ADX for trend strength and 1d RSI for overbought/oversold conditions
# Enter long when 4h ADX > 25 (trending) and 1d RSI < 40 (oversold bounce) with 1h close > 4h VWAP
# Enter short when 4h ADX > 25 (trending) and 1d RSI > 60 (overbought pullback) with 1h close < 4h VWAP
# Exit when trend weakens (ADX < 20) or RSI reverts to neutral range (40-60)
# Uses 4h/1d for directional bias and 1h for precise entry timing to avoid false signals
# Designed for low trade frequency (15-35/year) with trend-following logic that works in both bull and bear markets

name = "1h_ADX_RSI_VWAP_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX calculation (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) 
        # Subsequent values via Wilder smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_smooth = smooth_wilder(tr, period)
    dm_plus_smooth = smooth_wilder(dm_plus, period)
    dm_minus_smooth = smooth_wilder(dm_minus, period)
    
    # DI values
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, period)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(15, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 4h VWAP for 1h entry timing
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    vwap_4h = np.cumsum(typical_price_4h * volume_4h) / np.cumsum(volume_4h) if 'volume_4h' in df_4h else typical_price_4h
    if 'volume_4h' not in df_4h:
        volume_4h = np.ones_like(close_4h)  # fallback if volume not available
        vwap_4h = typical_price_4h
    else:
        volume_4h = df_4h['volume'].values
        vwap_4h = np.cumsum(typical_price_4h * volume_4h) / np.cumsum(volume_4h)
        # Handle division by zero
        vwap_4h = np.where(np.cumsum(volume_4h) != 0, vwap_4h, typical_price_4h)
    
    # Align HTF indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for all indicators
    start_idx = max(50, 200)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        adx_val = adx_4h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vwap_val = vwap_4h_aligned[i]
        close_val = close[i]
        
        if np.isnan(adx_val) or np.isnan(rsi_val) or np.isnan(vwap_val):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Trending market (ADX > 25), Oversold (RSI < 40), Price above VWAP
            if adx_val > 25 and rsi_val < 40 and close_val > vwap_val:
                signals[i] = 0.20
                position = 1
            # SHORT: Trending market (ADX > 25), Overbought (RSI > 60), Price below VWAP
            elif adx_val > 25 and rsi_val > 60 and close_val < vwap_val:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakening OR RSI reverting to neutral
            if adx_val < 20 or rsi_val > 50 or close_val < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Trend weakening OR RSI reverting to neutral
            if adx_val < 20 or rsi_val < 50 or close_val > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals