#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_v1"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA calculation (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Correct volatility calculation: sum of absolute changes over period
    volatility = pd.Series(close).rolling(window=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    ER = np.where(volatility != 0, change / volatility, 0)
    SC = (ER * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + SC[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for chop and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI < 40, chop > 61.8 (range), weekly uptrend
            if close[i] > kama[i] and rsi[i] < 40 and chop[i] > 61.8 and ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI > 60, chop > 61.8 (range), weekly downtrend
            elif close[i] < kama[i] and rsi[i] > 60 and chop[i] > 61.8 and ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or RSI > 60
            if close[i] < kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or RSI < 40
            if close[i] > kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA with RSI and Choppiness filter for mean reversion in ranging markets
# - KAMA adapts to market noise, providing dynamic support/resistance
# - In choppy markets (CHOP > 61.8), price tends to revert to KAMA
# - Long when price > KAMA, RSI < 40 (oversold), weekly uptrend
# - Short when price < KAMA, RSI > 60 (overbought), weekly downtrend
# - Exit when price crosses KAMA or RSI reaches opposite extreme
# - Works in both bull and bear markets via weekly trend filter
# - Choppiness filter ensures we only trade in ranging conditions
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Weekly EMA(34) trend filter prevents counter-trend trades in strong trends
# - Designed for low-frequency, high-probability mean reversion trades