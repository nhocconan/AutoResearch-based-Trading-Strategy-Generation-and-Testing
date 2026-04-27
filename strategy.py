#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week Bollinger Band breakout with 1-day RSI filter.
Breakouts occur when price moves beyond weekly Bollinger Bands (20, 2), filtered by 
daily RSI < 30 for longs and > 70 for shorts to capture mean reversion in overextended moves.
Volume > 1.5x average confirms breakout strength. Uses discrete position sizes (±0.25) 
to minimize fee churn. Target: 10-25 trades/year. Works in bull/bear by capturing 
overextended moves that revert to mean. Weekly timeframe reduces noise and false signals.
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
    
    # Get 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 1w data
    bb_period = 20
    close_1w = df_1w['close'].values
    
    # Middle band (SMA)
    sma_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= bb_period:
        for i in range(bb_period-1, len(close_1w)):
            sma_1w[i] = np.mean(close_1w[i-bb_period+1:i+1])
    
    # Standard deviation
    std_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= bb_period:
        for i in range(bb_period-1, len(close_1w)):
            std_1w[i] = np.std(close_1w[i-bb_period+1:i+1])
    
    # Upper and lower bands
    upper_bb = sma_1w + (2.0 * std_1w)
    lower_bb = sma_1w - (2.0 * std_1w)
    
    # Align Bollinger Bands to 1d timeframe (waits for 1w bar close)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Calculate RSI(14) on 1d data
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # Default to neutral
    
    for i in range(rsi_period-1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        elif avg_gain[i] > 0:  # Avoid division by zero
            rsi[i] = 100.0
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need BB (20), RSI (14), volume MA (20)
    start_idx = max(bb_period-1, rsi_period-1, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks below lower BB with oversold RSI and volume
            if price < lower_bb_aligned[i] and rsi[i] < 30 and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks above upper BB with overbought RSI and volume
            elif price > upper_bb_aligned[i] and rsi[i] > 70 and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle band or RSI normalizes
            if price > sma_1w_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to middle band or RSI normalizes
            if price < sma_1w_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_BollingerBreak_RSIMeanRev_Volume"
timeframe = "1d"
leverage = 1.0