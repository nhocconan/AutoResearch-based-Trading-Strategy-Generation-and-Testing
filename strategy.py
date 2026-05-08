#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI and volume spike
# Uses KAMA for adaptive trend following in trending markets and RSI for mean-reversion in ranging markets.
# Requires volume spike to confirm momentum and avoid false signals.
# Designed for low-frequency trades (<150 total) to minimize fee drag on 12h timeframe.

name = "12h_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute changes over ER period
    er_period = 10
    change_abs = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_sum = np.convolve(change_abs, np.ones(er_period), mode='same')
    volatility_sum[:er_period-1] = 0  # avoid invalid values
    er = np.where(volatility_sum != 0, change_abs / volatility_sum, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    sc[0] = 0
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # neutral
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume spike (2.0x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND RSI < 40 (mean reversion) AND volume spike
            if (close[i] > kama_12h_aligned[i] and 
                rsi_1d_aligned[i] < 40 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND RSI > 60 (mean reversion) AND volume spike
            elif (close[i] < kama_12h_aligned[i] and 
                  rsi_1d_aligned[i] > 60 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or RSI > 70 (overbought)
            if (close[i] < kama_12h_aligned[i] or 
                rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or RSI < 30 (oversold)
            if (close[i] > kama_12h_aligned[i] or 
                rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals