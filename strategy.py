#!/usr/bin/env python3
# 1h_Liquidity_Sweep_Retest_4hTrend_1dVolatility
# Hypothesis: In 1h timeframe, enter after liquidity sweeps (equal highs/lows) retest during 4h trend alignment and 1d low volatility.
# Works in bull/bear: liquidity sweeps occur in all regimes, trend filter ensures directionality, volatility filter avoids chop.
# Target: 20-35 trades/year via strict entry conditions.

name = "1h_Liquidity_Sweep_Retest_4hTrend_1dVolatility"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Equal highs/lows detection (liquidity pools) - 20 lookback
    lookback = 20
    equal_high = np.zeros(n, dtype=bool)
    equal_low = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Equal high: current high within 0.1% of lookback high
        lookback_high = np.max(high[i-lookback:i])
        if abs(high[i] - lookback_high) / lookback_high < 0.001:
            equal_high[i] = True
        
        # Equal low: current low within 0.1% of lookback low
        lookback_low = np.min(low[i-lookback:i])
        if abs(low[i] - lookback_low) / lookback_low < 0.001:
            equal_low[i] = True
    
    # Liquidity sweep detection: price breaks equal level then reverses
    liquidity_sweep_high = np.zeros(n, dtype=bool)  # swept high then closed below
    liquidity_sweep_low = np.zeros(n, dtype=bool)    # swept low then closed above
    
    for i in range(1, n):
        # Bullish sweep: swept equal low, then closed above it
        if equal_low[i] and low[i] < low[i-1] and close[i] > low[i]:
            liquidity_sweep_low[i] = True
        # Bearish sweep: swept equal high, then closed below it
        if equal_high[i] and high[i] > high[i-1] and close[i] < high[i]:
            liquidity_sweep_high[i] = True
    
    # 4h trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d volatility filter (ATR ratio low volatility)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) for 1d
    tr1 = np.zeros(len(high_1d))
    tr1[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]),
                     abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.zeros(len(high_1d))
    for i in range(13, len(tr1)):
        atr_1d[i] = np.mean(tr1[i-13:i+1])
    
    # Current ATR vs 20-period average (low volatility filter)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    low_volatility = atr_1d < atr_ma_1d  # ATR below average = low volatility
    
    # Align 1d volatility to 1h
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(low_volatility_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: liquidity sweep low retest, 4h uptrend, low volatility
            if (liquidity_sweep_low[i] and
                trend_4h_up_aligned[i] > 0.5 and
                low_volatility_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short: liquidity sweep high retest, 4h downtrend, low volatility
            elif (liquidity_sweep_high[i] and
                  trend_4h_down_aligned[i] > 0.5 and
                  low_volatility_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: liquidity sweep high or 4h trend turns down
            if (liquidity_sweep_high[i] or
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: liquidity sweep low or 4h trend turns up
            if (liquidity_sweep_low[i] or
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals