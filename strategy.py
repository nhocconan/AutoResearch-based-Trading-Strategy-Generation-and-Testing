# 1. Hypothesis: 4h strategy using 1-day KAMA trend with RSI mean reversion and volume confirmation.
# Long when KAMA turns up (bullish) and RSI < 40 (oversold) with volume > 1.5x average.
# Short when KAMA turns down (bearish) and RSI > 60 (overbought) with volume > 1.5x average.
# Exit when KAMA changes direction or RSI returns to neutral (40-60).
# Uses KAMA for adaptive trend, RSI for mean reversion, volume for confirmation.
# Target: 20-50 trades/year to avoid fee drag. Works in bull/bear via trend + mean reversion combo.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_fast=2, er_slow=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_slow:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_slow))  # |close[t] - close[t-er_slow]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |close[t] - close[t-1]| over er_slow period
    
    # Handle first er_slow elements
    er = np.full(n, np.nan)
    for i in range(er_slow-1, n):
        if volatility[i-er_slow+1:i+1].sum() > 0:
            er[i] = change[i-er_slow+1] / volatility[i-er_slow+1:i+1].sum()
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_slow-1] = np.mean(close[:er_slow])  # Start with SMA
    for i in range(er_slow, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 1d data
    kama_1d = calculate_kama(close_1d, er_fast=2, er_slow=30)
    
    # Calculate RSI on 4h data
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(prices), np.nan)
        avg_loss = np.full(len(prices), np.nan)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period+1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 30-period KAMA, 14-period RSI, 20-period volume MA
    start_idx = max(30, 14, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_now = rsi[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # KAMA direction (slope)
        if i > start_idx:
            kama_up = kama_aligned[i] > kama_aligned[i-1]
            kama_down = kama_aligned[i] < kama_aligned[i-1]
        else:
            kama_up = False
            kama_down = False
        
        if position == 0:
            # Long: KAMA up AND RSI oversold (<40) with volume confirmation
            if kama_up and rsi_now < 40 and vol_filter:
                signals[i] = size
                position = 1
            # Short: KAMA down AND RSI overbought (>60) with volume confirmation
            elif kama_down and rsi_now > 60 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns down OR RSI returns to neutral (>50)
            if not kama_up or rsi_now > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA turns up OR RSI returns to neutral (<50)
            if not kama_down or rsi_now < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_RSI_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0