#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly volatility filter and volume confirmation
# KAMA adapts to market noise, reducing false signals in choppy markets
# Weekly ATR ratio filters for low volatility environments where trends persist
# Volume > 1.3x 20-period average confirms institutional participation
# Designed for 1d timeframe with selective entries to avoid overtrading
# Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_ma = pd.Series(atr_1w).rolling(window=4, min_periods=4).mean().values  # 4-week MA
    atr_1w_ratio = atr_1w / atr_1w_ma  # Current ATR relative to 4-week average
    atr_1w_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_ratio)
    
    # Calculate KAMA on daily prices
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Pad beginning with zeros
    er = np.concatenate([np.zeros(10), er])
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(atr_1w_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine KAMA trend
        is_above_kama = close[i] > kama[i]
        is_below_kama = close[i] < kama[i]
        
        # Weekly volatility filter: low volatility environment (ATR ratio < 1.2)
        low_vol = atr_1w_ratio_aligned[i] < 1.2
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price above KAMA + low volatility + volume
            long_signal = is_above_kama and low_vol and has_volume
            
            # Short entry: price below KAMA + low volatility + volume
            short_signal = is_below_kama and low_vol and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or price crosses below KAMA
            stop_loss = entry_price - 2.5 * atr[i]
            kama_cross = price < kama[i]
            
            if stop_loss <= 0 or price <= stop_loss or kama_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or price crosses above KAMA
            stop_loss = entry_price + 2.5 * atr[i]
            kama_cross = price > kama[i]
            
            if stop_loss <= 0 or price >= stop_loss or kama_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_WeeklyVolFilter_Volume"
timeframe = "1d"
leverage = 1.0