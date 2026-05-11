#!/usr/bin/env python3
# 4H_KAMA_TREND_VOLUME_CONFIRM
# Hypothesis: KAMA (adaptive moving average) adapts to market noise, providing reliable trend signals.
# Combined with volume confirmation (>1.5x average volume) and 1d RSI filter (40-60) to avoid false signals.
# Designed for low-frequency trading (target: 20-40 trades/year) to minimize fee drag.
# Works in both bull and bear markets by following the adaptive trend.

name = "4H_KAMA_TREND_VOLUME_CONFIRM"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (10-period ER, 2 and 30 SC) ---
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Calculate ER (efficiency ratio) using rolling sum
    change_sum = pd.Series(change).rolling(window=10, min_periods=10).sum()
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum()
    er = change_sum / (volatility_sum + 1e-10)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- 1d RSI (14-period) ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align 1d RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if RSI is NaN
        if np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                # Exit if price crosses KAMA
                if position == 1 and close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: price crosses KAMA with volume confirmation and RSI in neutral zone
        long_entry = (close[i] > kama[i]) and vol_spike[i] and (rsi_1d_aligned[i] > 40) and (rsi_1d_aligned[i] < 60)
        short_entry = (close[i] < kama[i]) and vol_spike[i] and (rsi_1d_aligned[i] > 40) and (rsi_1d_aligned[i] < 60)
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on KAMA cross or RSI extreme
            if position == 1:
                if (close[i] < kama[i]) or (rsi_1d_aligned[i] >= 60) or (rsi_1d_aligned[i] <= 40):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > kama[i]) or (rsi_1d_aligned[i] >= 60) or (rsi_1d_aligned[i] <= 40):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals