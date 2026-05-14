#!/usr/bin/env python3
# Hypothesis: 12h KAMA trend direction + 1d RSI regime filter + 12h volume spike confirmation.
# Long when 12h KAMA is rising, 1d RSI < 30 (oversold bounce in trend), and 12h volume > 2.0x 20-period average.
# Short when 12h KAMA is falling, 1d RSI > 70 (overbought pullback in trend), and 12h volume > 2.0x 20-period average.
# Exit on opposite KAMA cross (falling for longs, rising for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. KAMA adapts to market noise,
# RSI filter avoids buying strength/selling weakness, volume confirms momentum.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Works in bull/bear: 1d RSI regime ensures mean-reversion within trend, KAMA filters whipsaws.

name = "12h_KAMA_Trend_1dRSI_Regime_12hVolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # 12h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (2.0 * vol_ma_20)
    
    # 12h KAMA (trend direction)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) >= 11 else np.zeros_like(change)
    volatility = np.concatenate([np.full(10, np.nan), volatility])  # align length
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    # Handle first element
    kama_rising[0] = False
    kama_falling[0] = False
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = np.concatenate([[np.nan], rsi_14])  # align length
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(kama[i]) or np.isnan(rsi_14_aligned[i]) or
            np.isnan(volume_confirm_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising + 1d RSI < 30 (oversold) + volume spike
            if (kama_rising[i] and 
                rsi_14_aligned[i] < 30 and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling + 1d RSI > 70 (overbought) + volume spike
            elif (kama_falling[i] and 
                  rsi_14_aligned[i] > 70 and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling
            if kama_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising
            if kama_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals