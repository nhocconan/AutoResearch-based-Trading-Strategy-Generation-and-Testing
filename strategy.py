#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    close_1d = df_1d['close'].values
    # Efficiency ratio calculation
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = np.sum(change[max(0, i-9):i+1]) / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    sc = (er * 0.6 + 0.06) ** 2  # smoothing constant
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Choppiness Index (CHOP)
    atr_1d = np.zeros(len(close_1d))
    tr1 = np.abs(np.subtract(df_1d['high'].values, df_1d['low'].values))
    tr2 = np.abs(np.subtract(df_1d['high'].values, np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr3 = np.abs(np.subtract(df_1d['low'].values, np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_atr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine trend from 1d KAMA
        uptrend = price > kama_aligned[i]
        downtrend = price < kama_aligned[i]
        
        # Chop regime: chop > 61.8 = ranging market
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price > KAMA, RSI < 30 (oversold), chop regime (mean reversion)
            if uptrend and rsi_aligned[i] < 30 and chop_regime:
                signals[i] = size
                position = 1
            # Short: price < KAMA, RSI > 70 (overbought), chop regime (mean reversion)
            elif downtrend and rsi_aligned[i] > 70 and chop_regime:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 50 (overbought) or trend breaks down
            if rsi_aligned[i] > 50 or price < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI < 50 (oversold) or trend breaks up
            if rsi_aligned[i] < 50 or price > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0