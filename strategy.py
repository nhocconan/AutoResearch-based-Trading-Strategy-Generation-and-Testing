#!/usr/bin/env python3
"""
4h Candlestick Momentum with Volume Spike and ADX Trend Filter
Hypothesis: Strong bullish/bearish candles with volume spikes signal momentum, but only in trending markets (ADX > 25).
In ranging markets (ADX < 20), we avoid trades to reduce false signals. Works in bull/bear by following momentum direction.
Uses volume confirmation to avoid weak moves and limits trades to reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth(val, p):
        result = np.full_like(val, np.nan)
        if len(val) < p:
            return result
        result[p-1] = np.mean(val[:p])
        for i in range(p, len(val)):
            result[i] = (result[i-1] * (p-1) + val[i]) / p
        return result
    
    atr = smooth(tr, period)
    plus_di = 100 * smooth(plus_dm, period) / atr
    minus_di = 100 * smooth(minus_dm, period) / atr
    dx = np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di)) * 100
    adx = smooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX for trend strength (1h timeframe for better signal)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    adx = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Candlestick momentum: strong bullish/bearish candle
    body_size = np.abs(close - open_)
    candle_range = high - low
    # Avoid division by zero
    body_ratio = np.where(candle_range > 0, body_size / candle_range, 0)
    
    open_ = prices['open'].values
    bullish_candle = (close > open_) & (body_ratio > 0.6)
    bearish_candle = (close < open_) & (body_ratio > 0.6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bullish_candle[i]) or np.isnan(bearish_candle[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish candle with volume spike in trending market (ADX > 25)
            if bullish_candle[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: bearish candle with volume spike in trending market (ADX > 25)
            elif bearish_candle[i] and vol_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish candle or ADX weakening
            if bearish_candle[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish candle or ADX weakening
            if bullish_candle[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Candlestick_Momentum_Volume_ADX"
timeframe = "4h"
leverage = 1.0