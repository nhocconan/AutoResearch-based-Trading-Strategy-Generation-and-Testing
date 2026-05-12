#!/usr/bin/env python3
# 1H_RSI_TREND_MOMENTUM
# Hypothesis: RSI(14) momentum combined with 4h EMA50 trend filter and volume confirmation.
# Long when RSI crosses above 50 from below with volume spike and price above 4h EMA50.
# Short when RSI crosses below 50 from above with volume spike and price below 4h EMA50.
# Exit when RSI returns to opposite side of 50 or trend invalidates.
# Uses 4h for trend direction, 1h for entry timing. Designed for 15-30 trades/year to minimize fee drag.

name = "1H_RSI_TREND_MOMENTUM"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 1.5
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    pclose_4h = df_4h['close'].values
    ema50_4h = pd.Series(pclose_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI crosses above 50 from below with volume spike and uptrend
            if rsi[i] > 50 and rsi[i-1] <= 50 and vol_spike[i] and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: RSI crosses below 50 from above with volume spike and downtrend
            elif rsi[i] < 50 and rsi[i-1] >= 50 and vol_spike[i] and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns below 50 or trend breaks
            if rsi[i] < 50 or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI returns above 50 or trend breaks
            if rsi[i] > 50 or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals