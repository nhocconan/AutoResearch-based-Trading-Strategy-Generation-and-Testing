#!/usr/bin/env python3
# 6H_RSI_MOMENTUM_DIVERGENCE
# Hypothesis: Combine 6-hour RSI momentum with 1-day trend filter to capture high-probability reversals in both bull and bear markets.
# Long when RSI(6) < 30 (oversold) and price above 1-day EMA50 (uptrend filter); short when RSI(6) > 70 (overbought) and price below 1-day EMA50 (downtrend filter).
# Exit when RSI returns to neutral zone (40-60) or trend invalidates.
# Uses volume confirmation to avoid false signals. Targets 15-25 trades/year per symbol to minimize fee drag.

name = "6H_RSI_MOMENTUM_DIVERGENCE"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (6-period) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma * 1.5
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold (<30) with volume confirmation and uptrend
            if rsi[i] < 30 and vol_filter[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) with volume confirmation and downtrend
            elif rsi[i] > 70 and vol_filter[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral or trend breaks
            if rsi[i] >= 40 or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral or trend breaks
            if rsi[i] <= 60 or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals