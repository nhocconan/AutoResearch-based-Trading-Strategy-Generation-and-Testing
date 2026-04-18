#!/usr/bin/env python3
"""
6h Moving Average Convergence Divergence (MACD) with Volume Confirmation and Volatility Filter
Hypothesis: MACD line crossing above/below signal line, with volume above 1.5x EMA(20) and 
price outside Bollinger Bands (20,2) indicates strong momentum with institutional participation.
Volatility filter (BB width > 0.05) avoids ranging markets. Designed for 6H timeframe to 
capture multi-day trends while minimizing false signals in both bull and bear markets.
Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # MACD components
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Bollinger Bands for volatility filter and entry confirmation
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * bb_std
    lower_bb = sma20 - 2 * bb_std
    bb_width = (upper_bb - lower_bb) / sma20  # Normalized bandwidth
    
    # Volume confirmation: volume > 1.5x EMA(20)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 26,20,20,9)
    
    for i in range(start_idx, n):
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
            np.isnan(bb_width[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        macd = macd_line[i]
        signal = signal_line[i]
        bb_w = bb_width[i]
        vol_conf = vol_ratio[i] > 1.5
        
        # Volatility filter: avoid extremely low volatility ranging markets
        vol_filter = bb_w > 0.05
        
        if position == 0:
            # Long: MACD crosses above signal line, price above upper BB, volume confirmation, adequate volatility
            if (macd > signal and macd_line[i-1] <= signal_line[i-1] and  # Bullish crossover
                price > upper_bb[i] and vol_conf and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: MACD crosses below signal line, price below lower BB, volume confirmation, adequate volatility
            elif (macd < signal and macd_line[i-1] >= signal_line[i-1] and  # Bearish crossover
                  price < lower_bb[i] and vol_conf and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: MACD crosses below signal line OR price returns to middle Bollinger Band
            if (macd < signal and macd_line[i-1] >= signal_line[i-1]) or price < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: MACD crosses above signal line OR price returns to middle Bollinger Band
            if (macd > signal and macd_line[i-1] <= signal_line[i-1]) or price > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MACD_Volume_BBFilter"
timeframe = "6h"
leverage = 1.0