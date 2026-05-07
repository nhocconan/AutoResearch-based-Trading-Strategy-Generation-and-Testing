#!/usr/bin/env python3
# 12h_1w_KAMA_Trend_1d_RSI_MeanReversion
# Hypothesis: Uses 1w KAMA for trend direction (bull/bear) and 1d RSI for mean-reversion entries.
# In bull market (price > 1w KAMA), look for RSI < 30 to go long; in bear market (price < 1w KAMA), look for RSI > 70 to go short.
# Combines trend-following with mean-reversion entries to work in both bull and bear markets.
# Uses volume confirmation to avoid false signals. Designed for 12h to keep trade frequency low (target: 15-35 trades/year).

name = "12h_1w_KAMA_Trend_1d_RSI_MeanReversion"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # Get 1w data for KAMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1w close
    # ER = |close - close_prev| / (sum |close - close_prev| over 10 periods)
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    er_1w = change_1w / (volatility_1w + 1e-10)  # Avoid division by zero
    # Smooth ER with EMA
    er_smooth_1w = pd.Series(er_1w).ewm(alpha=2/(10+1), adjust=False).fillna(0).values
    # Scaling factor: (ER * (fastest - slowest) + slowest)^2
    fastest = 2/(2+1)
    slowest = 2/(30+1)
    sc_1w = (er_smooth_1w * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama_1w = np.zeros_like(close_1w)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Get volume data for confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Align 1w KAMA and 1d RSI to 12h timeframe
    kama_1w_12h = align_htf_to_ltf(prices, df_1w, kama_1w)
    rsi_1d_12h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_1w_12h[i]) or np.isnan(rsi_1d_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bull trend: price > 1w KAMA -> look for RSI < 30 (oversold) for long
            if close[i] > kama_1w_12h[i] and rsi_1d_12h[i] < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Bear trend: price < 1w KAMA -> look for RSI > 70 (overbought) for short
            elif close[i] < kama_1w_12h[i] and rsi_1d_12h[i] > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 70 (overbought) or price < 1w KAMA (trend change)
            if rsi_1d_12h[i] > 70 or close[i] < kama_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 30 (oversold) or price > 1w KAMA (trend change)
            if rsi_1d_12h[i] < 30 or close[i] > kama_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals