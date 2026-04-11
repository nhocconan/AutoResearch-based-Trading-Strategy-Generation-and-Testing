#!/usr/bin/env python3
# 1d_1w_kama_rsi_volume_v1
# Strategy: 1d KAMA direction with RSI momentum filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI > 55 confirms bullish momentum, RSI < 45 confirms bearish momentum.
# Volume > 1.3x 20-period average confirms institutional participation.
# Weekly trend filter: only trade in direction of weekly KAMA to avoid counter-trend losses.
# Designed for low trade frequency (~10-20/year) to minimize fee drift.
# Works in bull markets via trend continuation and bear markets via counter-trend reversals at extremes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1d KAMA(10,2,30) - adaptive moving average
    def kama(close, er_fast=2, er_slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[-er_slow:])
        er = np.where(volatility > 0, change / volatility, 0)
        sc = np.power(er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = kama(close, 2, 30)
    
    # 1d RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly KAMA for trend filter
    kama_1w = kama(df_1w['close'].values, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or np.isnan(kama_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # KAMA direction: price above/below KAMA
        kama_bullish = close[i] > kama_1d[i]
        kama_bearish = close[i] < kama_1d[i]
        
        # Weekly trend filter: only trade in direction of weekly KAMA
        weekly_bullish = close[i] > kama_1w_aligned[i]
        weekly_bearish = close[i] < kama_1w_aligned[i]
        
        # RSI momentum filter: >55 for bullish, <45 for bearish
        rsi_bullish = rsi[i] > 55
        rsi_bearish = rsi[i] < 45
        
        # Entry conditions
        # Long: price > KAMA(1d) AND weekly bullish AND RSI bullish AND volume confirmation
        if kama_bullish and weekly_bullish and rsi_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price < KAMA(1d) AND weekly bearish AND RSI bearish AND volume confirmation
        elif kama_bearish and weekly_bearish and rsi_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA cross (trend change)
        elif position == 1 and not kama_bullish:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not kama_bearish:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals