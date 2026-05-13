#!/usr/bin/env python3
"""
4h_Volume_Weighted_RSI_Trend_Filter
Hypothesis: RSI(14) with volume-weighted smoothing and 1d trend filter captures momentum 
reversals in both bull and bear markets. Volume weighting reduces noise, making signals 
more reliable. Long when VW-RSI < 30 and price > 1d EMA50; short when VW-RSI > 70 and 
price < 1d EMA50. Exit on RSI reversion to 50 or trend change. Targets 20-40 trades/year.
"""

name = "4h_Volume_Weighted_RSI_Trend_Filter"
timeframe = "4h"
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
    
    # Volume-weighted RSI calculation
    def vwma(series, weights, period):
        """Volume Weighted Moving Average"""
        return np.convolve(series, weights, 'full')[:len(series)] / np.convolve(weights, np.ones_like(series), 'full')[:len(series)]
    
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted smoothing
    vol_weights = volume / np.mean(volume)  # Normalize volume
    vol_weights = np.where(vol_weights == 0, 1, vol_weights)  # Avoid division by zero
    
    # VWMA of gains and losses
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initialize with simple average for first period
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        # Wilder smoothing with volume weighting
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i] * vol_weights[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i] * vol_weights[i]) / 14
    
    # Calculate RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        rsi_val = rsi[i]
        vol_conf = volume_conf[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: VW-RSI oversold (<30) with 1d uptrend and volume confirmation
            if rsi_val < 30 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: VW-RSI overbought (>70) with 1d downtrend and volume confirmation
            elif rsi_val > 70 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to 50 or 1d trend turns down
            if rsi_val >= 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to 50 or 1d trend turns up
            if rsi_val <= 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals