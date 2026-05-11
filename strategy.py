#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_1dTrend_PriceAction
# Hypothesis: KAMA trend direction on 4h with RSI filter and 1d trend filter for low-frequency, high-conviction trades.
# Uses price action (higher highs/lows) to confirm trend and avoid whipsaws. Targets 20-30 trades/year.
# Designed to work in bull markets via trend following and bear markets via defensive positioning.

name = "4h_KAMA_Direction_RSI_1dTrend_PriceAction"
timeframe = "4h"
leverage = 1.0

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
    
    # === 4h KAMA (ER=10) for trend direction ===
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === RSI(14) on 4h ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Price Action: Higher Highs/Lows (3-bar lookback) ===
    hh = (high > np.maximum(high[1:], high[:-1])).astype(float)  # Simplified: current high > max of adjacent
    ll = (low < np.minimum(low[1:], low[:-1])).astype(float)
    # Pad arrays
    hh = np.concatenate([[hh[0]], hh])
    ll = np.concatenate([[ll[0]], ll])
    hh = hh[:n]
    ll = ll[:n]
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers KAMA, RSI, EMA)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(ema34_1d_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(hh[i]) or np.isnan(ll[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > KAMA (uptrend) + RSI > 50 + HH confirmation + above 1d EMA34
            if (close[i] > kama[i] and rsi[i] > 50 and hh[i] > 0 and 
                close[i] > ema34_1d_4h[i]):
                signals[i] = position_size
                position = 1
            # Short: Price < KAMA (downtrend) + RSI < 50 + LL confirmation + below 1d EMA34
            elif (close[i] < kama[i] and rsi[i] < 50 and ll[i] > 0 and 
                  close[i] < ema34_1d_4h[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses KAMA in opposite direction
            if position == 1:
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals