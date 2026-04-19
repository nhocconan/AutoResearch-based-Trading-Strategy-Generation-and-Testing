#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on weekly close
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan, dtype=np.float64)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14) on daily
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr, axis=1) / (highest_high - lowest_low)) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Chop > 61.8 = ranging market (mean revert)
        # Chop < 38.2 = trending market (trend follow)
        chop_threshold = 50.0  # neutral zone
        
        if position == 0:
            # Long: price > KAMA and RSI < 50 in ranging market, or RSI > 50 in trending
            if price > kama_val:
                if chop_val > chop_threshold:  # ranging
                    if rsi_val < 50:
                        signals[i] = 0.25
                        position = 1
                else:  # trending
                    if rsi_val > 50:
                        signals[i] = 0.25
                        position = 1
            # Short: price < KAMA and RSI > 50 in ranging, or RSI < 50 in trending
            elif price < kama_val:
                if chop_val > chop_threshold:  # ranging
                    if rsi_val > 50:
                        signals[i] = -0.25
                        position = -1
                else:  # trending
                    if rsi_val < 50:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA
            if price < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA
            if price > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals