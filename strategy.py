#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend following with 1d RSI filter and volume confirmation
# Long when KAMA trend is up AND 1d RSI < 50 (avoid overbought) AND volume > 1.3x average
# Short when KAMA trend is down AND 1d RSI > 50 (avoid oversold) AND volume > 1.3x average
# Exit when KAMA trend reverses or opposite signal occurs
# KAMA adapts to market noise, reducing whipsaw in sideways markets. RSI filter prevents
# entries at extremes. Volume confirms institutional interest. Designed for both bull/bear.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on close (ER=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation properly
    volatility_series = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum()
    er = change / (volatility_series + 1e-10)  # Efficiency Ratio
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Smoothing Constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI (14-period)
    delta = np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 1d RSI to 4h timeframe with proper delay (wait for daily close)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi_aligned[i]
        close_val = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.3
        
        if position == 0:
            # Long setup: KAMA trending up AND RSI < 50 (not overbought) AND volume confirmation
            if (kama_val > kama_prev and rsi_val < 50 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA trending down AND RSI > 50 (not oversold) AND volume confirmation
            elif (kama_val < kama_prev and rsi_val > 50 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA trend reverses down OR opposite signal
            if (kama_val < kama_prev or 
                (rsi_val > 70 and vol > vol_threshold)):  # Exit on overbought with volume
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA trend reverses up OR opposite signal
            if (kama_val > kama_prev or 
                (rsi_val < 30 and vol > vol_threshold)):  # Exit on oversold with volume
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0