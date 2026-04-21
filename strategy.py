#!/usr/bin/env python3
"""
6h_RSI_Divergence_With_Volume_Filter
Hypothesis: Combines RSI divergence (bullish/bearish) with volume confirmation on 6h timeframe.
Uses 1d ADX as trend filter to avoid counter-trend trades in strong trends.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in both bull and bear markets by trading mean reversion in ranging markets and
pullbacks in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d ADX(14) trend filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    adx_1d = adx
    
    # === 6h RSI(14) ===
    close = prices['close'].values
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # Wilder's smoothing for RSI
    avg_gain[13] = np.nanmean(gain[1:14])
    avg_loss[13] = np.nanmean(loss[1:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h Volume Average (20-period) ===
    volume = prices['volume'].values
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.nanmean(volume[i-19:i+1])
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for RSI divergence
        bullish_div = False
        bearish_div = False
        
        # Look back 5 periods for divergence
        lookback = 5
        if i >= lookback:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (close[i] < close[i-lookback] and 
                rsi[i] > rsi[i-lookback]):
                # Confirm with higher low in price
                if np.nanmin(close[i-lookback:i+1]) == close[i]:
                    bullish_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (close[i] > close[i-lookback] and 
                rsi[i] < rsi[i-lookback]):
                # Confirm with lower high in price
                if np.nanmax(close[i-lookback:i+1]) == close[i]:
                    bearish_div = True
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # ADX filter: only trade when ADX < 25 (ranging market) or 
        # when ADX > 25 and divergence aligns with trend
        adx_val = adx_1d_aligned[i]
        
        if position == 0:
            # Long conditions: bullish divergence + volume + (ADX<25 or ADX>25 with bullish bias)
            if bullish_div and volume_confirm:
                if adx_val < 25 or (adx_val >= 25 and rsi[i] < 50):  # In trend, look for pullback
                    signals[i] = 0.25
                    position = 1
            # Short conditions: bearish divergence + volume + (ADX<25 or ADX>25 with bearish bias)
            elif bearish_div and volume_confirm:
                if adx_val < 25 or (adx_val >= 25 and rsi[i] > 50):  # In trend, look for pullback
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long on bearish divergence or RSI > 70 (overbought)
                if bearish_div and volume_confirm or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short on bullish divergence or RSI < 30 (oversold)
                if bullish_div and volume_confirm or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_RSI_Divergence_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0