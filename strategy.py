#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA trend filter from 1d (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(change) > 0 else 0
    # Vectorized ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            ch = np.abs(close_1d[i] - close_1d[i-10])
            vol = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = ch / vol if vol != 0 else 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on 4h for mean reversion signals
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value to match length
    rsi = np.concatenate([[50], rsi])
    
    # Bollinger Bands width for regime filter (chop detection)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    # Chop regime: high BB width = trending, low BB width = ranging
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    chop_threshold = 0.5  # Below 0.5 = choppy/ranging, above = trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_width_rank[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend) + oversold RSI + choppy market
            long_cond = (close[i] < kama_1d_aligned[i]) and \
                        (rsi[i] < 30) and \
                        (bb_width_rank[i] < chop_threshold)
            # Short: price above KAMA (rally in downtrend) + overbought RSI + choppy market
            short_cond = (close[i] > kama_1d_aligned[i]) and \
                         (rsi[i] > 70) and \
                         (bb_width_rank[i] < chop_threshold)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above KAMA or RSI overbought
            if close[i] > kama_1d_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below KAMA or RSI oversold
            if close[i] < kama_1d_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals