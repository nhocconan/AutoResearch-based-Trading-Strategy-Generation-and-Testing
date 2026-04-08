#!/usr/bin/env python3
# 12h_kama_rsi_volatility
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# Long when KAMA direction is up, RSI < 30 (oversold), and volatility (ATR ratio) < 1.0 (low volatility).
# Short when KAMA direction is down, RSI > 70 (overbought), and volatility (ATR ratio) < 1.0.
# Exit when RSI crosses 50 (mean reversion signal).
# Designed to capture mean reversion in low-volatility environments with adaptive trend filter.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_volatility"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA on 12h data (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    
    # Calculate RSI on 12h data (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR on daily data (14-period) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Normalize ATR by price to get volatility ratio
    atr_ratio = atr / close_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volatility filter: low volatility (ATR ratio < 1.0)
            vol_ok = atr_ratio_aligned[i] < 1.0
            
            # Mean reversion entries: KAMA direction + RSI extremes
            if (kama_dir[i] == 1) and (rsi[i] < 30) and vol_ok:
                position = 1
                signals[i] = 0.25
            elif (kama_dir[i] == -1) and (rsi[i] > 70) and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals