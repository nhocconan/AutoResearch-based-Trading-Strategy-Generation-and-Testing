#!/usr/bin/env python3
# 4h_RSI_Extreme_Trend_Follow_Volume
# Hypothesis: Use RSI extremes combined with 1d trend filter and volume confirmation to capture strong momentum moves in both bull and bear markets.
# Only trade when RSI shows extreme conditions (oversold in uptrend, overbought in downtrend) to avoid chop.
# This reduces whipsaws and focuses on high-probability trend continuation moves.
# Designed for moderate trade frequency (target: 20-50 trades/year) with clear entry/exit rules.

name = "4h_RSI_Extreme_Trend_Follow_Volume"
timeframe = "4h"
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
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        trend_up = close_1d_current > ema50_1d_aligned[i]
        trend_down = close_1d_current < ema50_1d_aligned[i]
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: RSI < 30 (oversold) in uptrend with volume confirmation
            if rsi[i] < 30 and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) in downtrend with volume confirmation
            elif rsi[i] > 70 and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 (overbought) or trend breakdown
            if rsi[i] > 70 or close_1d_current < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 30 (oversold) or trend reversal
            if rsi[i] < 30 or close_1d_current > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals