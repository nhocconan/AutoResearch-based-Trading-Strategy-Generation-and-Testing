#!/usr/bin/env python3
# 1D_RSI_EMA200_Downtrend_Pullback
# Hypothesis: Mean-reversion pullback in downtrend using daily timeframe. 
# Enters long when RSI < 30 (oversold) and price > EMA200 (avoid downtrend).
# Exits when RSI > 50 (mean reversion) or price < EMA200 (trend resumption).
# Uses volume confirmation (1.5x avg) to filter false signals.
# Designed for low trade frequency (<20/year) to minimize fee drag and work in both bull/bear regimes.

name = "1D_RSI_EMA200_Downtrend_Pullback"
timeframe = "1d"
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
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA200 for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = max(200, 20)  # Ensure EMA200 and volume MA are valid
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema200[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30), price above EMA200 (avoid strong downtrend), volume confirmation
            if (rsi[i] < 30 and 
                close[i] > ema200[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit: RSI > 50 (mean reversion complete) or price < EMA200 (trend resumption)
            if (rsi[i] > 50 or 
                close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals