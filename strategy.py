#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-day KAMA trend direction + 1-day RSI overbought/oversold + 1-day volatility regime.
KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI extremes with trend filter capture mean reversions in strong trends.
Volatility filter (ATR ratio < 1.0) avoids trading in extremely high volatility where false signals prevail.
Targets 50-150 total trades over 4 years to minimize fee drag. Works in bull/bear via adaptive trend + mean reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for KAMA, RSI, ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # ER = |Change| / Volatility
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # placeholder, will fix below
    # Correct volatility calculation: sum of absolute changes over period
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    # Actually, use rolling sum for efficiency
    volatility = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smooth constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ATR(14) for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    
    # Volatility regime: ATR ratio < 1.0 = normal volatility (avoid extreme volatility)
    normal_volatility = atr_ratio < 1.0
    
    # Align all indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    normal_volatility_aligned = align_htf_to_ltf(prices, df_1d, normal_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(normal_volatility_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price > KAMA (uptrend) AND RSI < 30 (oversold) AND normal volatility
        long_entry = (close[i] > kama_aligned[i]) and (rsi_aligned[i] < 30) and (normal_volatility_aligned[i] > 0.5)
        # Short: price < KAMA (downtrend) AND RSI > 70 (overbought) AND normal volatility
        short_entry = (close[i] < kama_aligned[i]) and (rsi_aligned[i] > 70) and (normal_volatility_aligned[i] > 0.5)
        
        # Exit when price crosses KAMA in opposite direction (trend change)
        exit_long = position == 1 and close[i] < kama_aligned[i]
        exit_short = position == -1 and close[i] > kama_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_kama_rsi_vol"
timeframe = "12h"
leverage = 1.0