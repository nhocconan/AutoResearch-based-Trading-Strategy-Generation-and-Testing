#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_MACD_Trend_12hVWAP_Divergence"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # MACD on 6h: fast=12, slow=26, signal=9
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close_s.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - macd_signal
    
    # 12h VWAP for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_numerator = np.cumsum(typical_price_12h * volume_12h)
    vwap_denominator = np.cumsum(volume_12h)
    vwap_12h = vwap_numerator / vwap_denominator
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Volume confirmation: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(macd_hist[i]) or np.isnan(macd_signal[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD bullish crossover + price above VWAP + volume
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 and  # bullish crossover
                close[i] > vwap_12h_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: MACD bearish crossover + price below VWAP + volume
            elif (macd_hist[i] < 0 and macd_hist[i-1] >= 0 and  # bearish crossover
                  close[i] < vwap_12h_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: MACD bearish crossover or price below VWAP
            if (macd_hist[i] < 0 and macd_hist[i-1] >= 0 or  # bearish crossover
                close[i] < vwap_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: MACD bullish crossover or price above VWAP
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 or  # bullish crossover
                close[i] > vwap_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals