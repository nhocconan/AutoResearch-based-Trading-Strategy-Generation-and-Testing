#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_Volume_Confirmation
Hypothesis: In both bull and bear markets, RSI extremes on 4h timeframe combined with volume spike and price reversal from Bollinger Bands provides mean-reversion opportunities. Uses volume confirmation to filter false signals and Bollinger Band width to identify high-probability reversal zones. Designed for lower trade frequency (<50/year) to minimize fee drag.
"""

name = "4h_RSI_MeanReversion_Volume_Confirmation"
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
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = ma20 + (2 * std20)
    lower_band = ma20 - (2 * std20)
    
    # Bollinger Band Width for regime filter
    bb_width = (upper_band - lower_band) / ma20
    bb_width_ma50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for BB width MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ma20[i]) or 
            np.isnan(std20[i]) or 
            np.isnan(vol_ratio[i]) or 
            np.isnan(bb_width_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when volatility is contracting (low BB width)
        # This identifies consolidation periods before mean reversion
        if bb_width[i] > bb_width_ma50[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: RSI oversold (<30) + price at/below lower BB + volume spike
            if (rsi[i] < 30 and 
                close[i] <= lower_band[i] and 
                vol_ratio[i] > 2.5):
                signals[i] = 0.25
                position = 1
            # Short signal: RSI overbought (>70) + price at/above upper BB + volume spike
            elif (rsi[i] > 70 and 
                  close[i] >= upper_band[i] and 
                  vol_ratio[i] > 2.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or price reaches middle band
            if rsi[i] >= 50 or close[i] >= ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or price reaches middle band
            if rsi[i] <= 50 or close[i] <= ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals