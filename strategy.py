#!/usr/bin/env python3
"""
4h RSI Trend Reversal with Volume Spike and ADX Filter
Hypothesis: RSI extremes (oversold/overbought) combined with volume spikes indicate 
potential reversals. ADX > 25 ensures we trade in trending markets, avoiding 
whipsaws in sideways conditions. Works in bull markets (buy dips) and bear markets 
(sell rallies) by fading extreme RSI readings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period * 2:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothing
    atr = np.zeros_like(high)
    atr[period-1] = np.mean(tr[:period])
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
        minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
    
    # DX and ADX
    dx = np.zeros_like(high)
    adx = np.zeros_like(high)
    
    for i in range(period, len(high)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / di_sum * 100
        else:
            dx[i] = 0
    
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI for overbought/oversold signals
    rsi = calculate_rsi(close, 14)
    
    # ADX for trend strength filter
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long signal: RSI oversold (<30) + volume spike + trending market (ADX > 25)
            if (rsi[i] < 30 and 
                vol_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short signal: RSI overbought (>70) + volume spike + trending market (ADX > 25)
            elif (rsi[i] > 70 and 
                  vol_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or volume spike ends
            if rsi[i] > 50 or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or volume spike ends
            if rsi[i] < 50 or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Reversal_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0