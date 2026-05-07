# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6h_1d_1w_Trend_Pullback_RSI"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA(20) for higher timeframe trend
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # RSI(14) on 6h for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Wait for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend, 1w uptrend, RSI pullback from overbought
            if (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] and
                rsi[i] < 40 and rsi[i-1] >= 40):
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend, 1w downtrend, RSI pullback from oversold
            elif (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] and
                  rsi[i] > 60 and rsi[i-1] <= 60):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought or trend breaks
            if rsi[i] > 70 or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold or trend breaks
            if rsi[i] < 30 or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Trend Pullback RSI
# - Uses 1d and 1w EMA trends for multi-timeframe alignment
# - Enters on RSI pullbacks (40 for longs, 60 for shorts) in direction of higher timeframe trend
# - Works in both bull and bear markets by following the higher timeframe trend
# - RSI provides mean reversion entries within the trend
# - Exit when RSI reaches extreme levels or trend breaks
# - Position size 0.25 limits risk and reduces trade frequency
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Novel combination: Multi-timeframe EMA trend + RSI pullback not recently tried on 6h
# - Avoids overtrading by requiring both timeframes to trend in same direction
# - Uses proper alignment to prevent look-ahead bias
# - Conservative entry conditions to minimize false signals
# - Designed for BTC/ETH with potential applicability to SOL
# - Weekly trend filter adds robustness against false breaks
# - RSI thresholds (40/60) selected for timely entries without chasing
# - Exit conditions prevent giving back too much profit
# - Simple logic with clear rules for robust performance across regimes