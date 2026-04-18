#!/usr/bin/env python3
"""
4h_RSI_40_60_MeanReversion_Range
Hypothesis: Mean reversion strategy using RSI extremes (40/60) with volume confirmation and Bollinger Band squeeze filter.
Designed for low trade frequency (target: 20-50/year) in both bull and bear markets by avoiding overtrading.
Uses RSI as primary signal, volume spike for confirmation, and Bollinger Band width percentile to filter ranging markets.
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
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    bb_middle = np.full(n, np.nan)
    bb_std = np.full(n, np.nan)
    for i in range(20, n):
        bb_middle[i] = np.mean(close[i-20:i])
        bb_std[i] = np.std(close[i-20:i])
    
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (252-period for 1-year lookback)
    bb_width_percentile = np.full(n, np.nan)
    for i in range(252, n):
        window = bb_width[i-252:i]
        if not np.all(np.isnan(window)):
            bb_width_percentile[i] = (np.sum(bb_width[i] > window) / np.sum(~np.isnan(window))) * 100
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 252)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 40 (oversold) with volume spike and low volatility (range)
            if (rsi[i] < 40 and vol_spike[i] and bb_width_percentile[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (overbought) with volume spike and low volatility (range)
            elif (rsi[i] > 60 and vol_spike[i] and bb_width_percentile[i] < 30):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or volatility increases
            if (rsi[i] > 50 or bb_width_percentile[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or volatility increases
            if (rsi[i] < 50 or bb_width_percentile[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_40_60_MeanReversion_Range"
timeframe = "4h"
leverage = 1.0