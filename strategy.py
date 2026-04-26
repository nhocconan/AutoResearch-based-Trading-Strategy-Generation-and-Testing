#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering. Long when KAMA upward, RSI > 50, and CHOP < 38.2 (trending regime). Short when KAMA downward, RSI < 50, and CHOP < 38.2. Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets by only taking trades in strong trending regimes (CHOP < 38.2) and following the adaptive trend. Weekly HTF trend filter (1w EMA50) ensures alignment with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for KAMA, RSI, CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for HTF trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA calculation (ER=10, fastest=2, slowest=30)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10)).values
    volatility = np.abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(atr_sum != 0, 100 * np.log10(max_high - min_low) / np.log10(atr_sum) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA, 14 for RSI/CHOP)
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Get current values
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_1w_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(ema_1w_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # KAMA trend direction
        kama_up = kama_val > kama_prev
        kama_down = kama_val < kama_prev
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        # Long logic: KAMA up, RSI > 50, trending regime, and above weekly EMA (bullish HTF)
        long_condition = kama_up and (rsi_val > 50) and trending_regime and (close[i] > ema_1w_val)
        # Short logic: KAMA down, RSI < 50, trending regime, and below weekly EMA (bearish HTF)
        short_condition = kama_down and (rsi_val < 50) and trending_regime and (close[i] < ema_1w_val)
        
        # Exit logic: reverse conditions or regime change to choppy
        long_exit = (position == 1 and (not kama_up or rsi_val < 50 or chop_val >= 38.2))
        short_exit = (position == -1 and (not kama_down or rsi_val > 50 or chop_val >= 38.2))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0