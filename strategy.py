#!/usr/bin/env python3
name = "6h_Adaptive_Kelly_Volume_Regime_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import exp, sqrt
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and volatility
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA50 for trend
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h ATR for volatility
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(abs(high_12h - np.roll(close_12h, 1)),
                                   abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 6h RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    # Kelly fraction calculation (simplified)
    # Win probability based on RSI extremes
    prob_win_long = np.where(rsi < 30, 0.6,
                     np.where(rsi > 70, 0.4, 0.5))
    prob_win_short = np.where(rsi > 70, 0.6,
                      np.where(rsi < 30, 0.4, 0.5))
    
    # Win/loss ratio based on volatility
    win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk
    
    # Kelly f = (bp - q) / b where b = win_loss_ratio, p = prob_win, q = prob_loss
    kelly_long = (win_loss_ratio * prob_win_long - (1 - prob_win_long)) / win_loss_ratio
    kelly_short = (win_loss_ratio * prob_win_short - (1 - prob_win_short)) / win_loss_ratio
    
    # Cap Kelly at 0.3 and apply volatility scaling
    vol_normalized = atr_12h_aligned / (pd.Series(atr_12h_aligned).rolling(50, min_periods=1).mean().values + 1e-10)
    vol_scaling = np.clip(1.0 / vol_normalized, 0.5, 2.0)  # Inverse vol scaling
    
    kelly_long = np.clip(kelly_long, 0, 0.3) * vol_scaling
    kelly_short = np.clip(kelly_short, 0, 0.3) * vol_scaling
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ok[i]) or
            np.isnan(kelly_long[i]) or np.isnan(kelly_short[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold + above 12h EMA (uptrend) + volume
            if rsi[i] < 30 and close[i] > ema_12h_aligned[i] and volume_ok[i]:
                signals[i] = kelly_long[i]
                position = 1
            # Short: RSI overbought + below 12h EMA (downtrend) + volume
            elif rsi[i] > 70 and close[i] < ema_12h_aligned[i] and volume_ok[i]:
                signals[i] = -kelly_short[i]
                position = -1
        elif position == 1:
            # Long exit: RSI overbought OR below 12h EMA (trend change)
            if rsi[i] > 70 or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = kelly_long[i]  # maintain position
        elif position == -1:
            # Short exit: RSI oversold OR above 12h EMA (trend change)
            if rsi[i] < 30 or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -kelly_short[i]  # maintain position
    
    return signals