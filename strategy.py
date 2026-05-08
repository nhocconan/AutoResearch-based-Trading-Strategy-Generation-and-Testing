#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA trend with daily RSI filter and volume confirmation
# Long when KAMA rises above price (uptrend), RSI > 50, volume spike
# Short when KAMA falls below price (downtrend), RSI < 50, volume spike
# KAMA adapts to market noise, reducing false signals in choppy markets
# RSI filter ensures momentum alignment with trend
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_KAMA_DailyRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA(10,2,30): fast=2, slow=30
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    er = change / (volatility + 1e-10)
    # Smoothing Constant
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_1d_aligned[i]
        kama_val = kama[i]
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: KAMA above price (uptrend), RSI > 50, volume spike
            if kama_val > close_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA below price (downtrend), RSI < 50, volume spike
            elif kama_val < close_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falls below price or RSI < 50
            if kama_val < close_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rises above price or RSI > 50
            if kama_val > close_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals