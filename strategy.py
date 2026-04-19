#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points for R1 and S1
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align daily pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Momentum filter: RSI(14) between 40 and 60 to avoid extremes
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_values[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        rsi_filter = 40 <= rsi_val <= 60
        
        if position == 0:
            # Long: Price breaks above R1 with volume and RSI filter
            if price > r1_4h[i] and volume_confirmed and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and RSI filter
            elif price < s1_4h[i] and volume_confirmed and rsi_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot
            if price < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot
            if price > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals