#!/usr/bin/env python3
name = "1d_1w_WeeklyKAMA_RSI_Trend"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SM = Smoothing Constant
    change = np.abs(df_1w['close'].diff(10).values)
    volatility = np.abs(df_1w['close'].diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(df_1w['close'])
    kama[0] = df_1w['close'].iloc[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1w['close'].iloc[i] - kama[i-1])
    
    # Weekly RSI(14)
    delta = df_1w['close'].diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Daily volume spike detection: 3-day average
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA, RSI > 50 (bullish), volume spike
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: price below weekly KAMA, RSI < 50 (bearish), volume spike
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below KAMA or RSI turns bearish
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above KAMA or RSI turns bullish
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 1d KAMA/RSI trend following with weekly trend filter and volume confirmation
# - Weekly KAMA acts as dynamic trend filter (adapts to market conditions)
# - Weekly RSI > 50 confirms bullish momentum, < 50 confirms bearish momentum
# - Daily price position relative to weekly KAMA determines entry direction
# - Volume spike (2x 3-day average) confirms institutional participation
# - Works in both bull (buy when price > KAMA in weekly uptrend) and bear (sell when price < KAMA in weekly downtrend)
# - Exit when price crosses back below/above KAMA or RSI reverses
# - Position size 0.30 targets ~20-40 trades/year, minimizing fee drag
# - Weekly timeframe reduces noise and whipsaws compared to lower timeframes
# - Designed to capture major trends while avoiding choppy markets via RSI filter