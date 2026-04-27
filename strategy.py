#!/usr/bin/env python3
"""
4h_KAMA_Trend_1dRSI_VolumeFilter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
Combined with 1d RSI for overbought/oversold conditions and volume confirmation,
this strategy aims to capture trends while avoiding whipsaws in ranging markets.
Designed to work in both bull and bear markets by using adaptive trend filtering.
Target: 20-50 trades/year to minimize fee drag.
"""

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
    
    # Get 4h data for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate KAMA(10,2,30) on 4h close
    close_4h = df_4h['close'].values
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    kama = np.full(len(close_4h), np.nan)
    if len(close_4h) >= er_period:
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close_4h, er_period))  # |close[i] - close[i-er_period]|
        volatility = np.sum(np.abs(np.diff(close_4h)), axis=1)  # sum of |diff| over er_period window
        # Handle first er_period elements
        change = np.concatenate([np.full(er_period, np.nan), change])
        volatility = np.concatenate([np.full(er_period, np.nan), volatility])
        er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
        sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
        # Initialize KAMA
        kama[er_period] = close_4h[er_period]
        for i in range(er_period + 1, len(close_4h)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d close
    close_1d = df_1d['close'].values
    rsi_period = 14
    rsi = np.full(len(close_1d), np.nan)
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation on 4h
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(er_period, rsi_period, vol_ma_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price relative to KAMA
        above_kama = price > kama_aligned[i]
        below_kama = price < kama_aligned[i]
        
        # RSI conditions: avoid extremes
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Volume confirmation: > 1.3x average volume
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long: price above KAMA, not overbought, volume confirmation
            if above_kama and rsi_not_overbought and volume_confirmation:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, not oversold, volume confirmation
            elif below_kama and rsi_not_oversold and volume_confirmation:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI overbought
            if price < kama_aligned[i] or rsi_aligned[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI oversold
            if price > kama_aligned[i] or rsi_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_KAMA_Trend_1dRSI_VolumeFilter"
timeframe = "4h"
leverage = 1.0