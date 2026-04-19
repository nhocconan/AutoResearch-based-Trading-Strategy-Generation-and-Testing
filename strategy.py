#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w KAMA (adaptive moving average)
    # Efficiency ratio: price change / volatility
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close_1w, k=1)), axis=0)  # 10-period volatility
    # Pad arrays to match length
    change = np.concatenate([[np.nan]*10, change])
    vol = np.concatenate([[np.nan]*10, vol])
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate 1w RSI(14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.concatenate([[np.nan]*13, [np.mean(gain[1:14])]])
    avg_loss = np.concatenate([[np.nan]*13, [np.mean(loss[1:14])]])
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1w Choppiness Index(14)
    # True range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of true range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Daily ATR(14) for volatility filter
    tr_d1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr_d2 = np.abs(high[1:] - close[:-1])
    tr_d3 = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr_d1, np.maximum(tr_d2, tr_d3))])
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(chop_aligned[i]) or np.isnan(atr_d[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        atr = atr_d[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr > 0.01 * price  # ATR > 1% of price
        
        # Chop regime filter: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        chop_ranging = chop_val > 61.8
        chop_trending = chop_val < 38.2
        
        if position == 0:
            # Long: price above KAMA, RSI oversold in ranging market
            if price > kama_val and rsi_val < 30 and chop_ranging and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI overbought in ranging market
            elif price < kama_val and rsi_val > 70 and chop_ranging and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals