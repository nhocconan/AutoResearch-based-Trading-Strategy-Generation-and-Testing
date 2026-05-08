#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1w RSI and 1d volume confirmation.
# Long when KAMA trend is up AND 1w RSI < 40 (oversold) AND 1d volume > 1.2x 20-period average.
# Short when KAMA trend is down AND 1w RSI > 60 (overbought) AND 1d volume > 1.2x 20-period average.
# Exit when KAMA trend reverses.
# Uses 12h timeframe with 1w RSI for regime and 1d volume for confirmation.
# Target: 50-150 total trades over 4 years with controlled frequency to avoid fee drag.

name = "12h_KAMA_1wRSI_1dVolume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 12h data
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]).reshape(-1, 1)), axis=1)[:len(close)]
    volatility = pd.Series(volatility).rolling(window=er_period, min_periods=1).sum().values
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_up = kama > np.roll(kama, 1)
    kama_up[0] = False
    
    # Weekly RSI(14)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    delta = np.diff(close_w, prepend=close_w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_w = 100 - (100 / (1 + rs))
    rsi_w[np.isnan(rsi_w)] = 50
    
    # Align weekly RSI to 12h timeframe
    rsi_w_aligned = align_htf_to_ltf(prices, df_w, rsi_w)
    
    # Daily volume filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.2 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up, RSI oversold, volume confirmation
            long_cond = kama_up[i] and (rsi_w_aligned[i] < 40) and volume_filter[i]
            # Short conditions: KAMA down, RSI overbought, volume confirmation
            short_cond = (not kama_up[i]) and (rsi_w_aligned[i] > 60) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA trend turns down
            if not kama_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA trend turns up
            if kama_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals