#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_Filter_Trend_Follow
# Hypothesis: KAMA adapts to market noise, providing a smooth trend line that lags less in strong trends and filters noise in chop.
# Combined with RSI(14) to avoid overextended entries and a 4h EMA50 trend filter to ensure alignment with medium-term trend.
# Works in bull markets via longs above KAMA and EMA50 with RSI<70, and in bear markets via shorts below KAMA and EMA50 with RSI>30.
# Uses volume confirmation to avoid false breakouts. Designed for fewer, higher-quality trades.

name = "4h_KAMA_Direction_RSI_Filter_Trend_Follow"
timeframe = "4h"
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
    
    # === KAMA Calculation ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Pad the beginning with NaN for alignment
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after first 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    # First average (simple mean)
    avg_gain[14] = np.mean(gain[1:15])  # gains from index 1 to 14
    avg_loss[14] = np.mean(loss[1:15])
    # Wilder smoothing
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h EMA50 Trend Filter ===
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and other indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA and EMA50, RSI not overbought, volume spike
            if close[i] > kama[i] and close[i] > ema_50[i] and rsi[i] < 70 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA and EMA50, RSI not oversold, volume spike
            elif close[i] < kama[i] and close[i] < ema_50[i] and rsi[i] > 30 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA or EMA50
            if close[i] < kama[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or EMA50
            if close[i] > kama[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals