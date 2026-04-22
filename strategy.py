#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly Hull Moving Average filter and volume confirmation
# Long when KAMA crosses above HMA + volume > 1.5x average
# Short when KAMA crosses below HMA + volume > 1.5x average
# Exit when KAMA crosses back across HMA
# Designed for low trade frequency (<10/year) to minimize fee drag while capturing major trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Hull Moving Average trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Hull Moving Average (HMA)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    n_hull = 16
    half_n = n_hull // 2
    sqrt_n = int(np.sqrt(n_hull))
    
    def wma(arr, window):
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='same')
    
    wma_full = wma(close_weekly, n_hull)
    wma_half = wma(close_weekly, half_n)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    hma_aligned = align_htf_to_ltf(prices, df_weekly, hma)
    
    # Load daily data for KAMA trend
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA)
    # ER = |Change| / Volatility
    # SC = [ER * (fastest - slowest) + slowest]^2
    change = np.abs(np.diff(close_daily, k=10))
    volatility = np.sum(np.abs(np.diff(close_daily, k=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    fastest = 2 / (2 + 1)
    slowest = 2 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i-1] * (close_daily[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(hma_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_aligned[i]
        hma_val = hma_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: KAMA crosses above HMA + volume filter
            if kama_val > hma_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA crosses below HMA + volume filter
            elif kama_val < hma_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when KAMA crosses back across HMA
            if (position == 1 and kama_val < hma_val) or \
               (position == -1 and kama_val > hma_val):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_HMA_Volume_Filter"
timeframe = "1d"
leverage = 1.0