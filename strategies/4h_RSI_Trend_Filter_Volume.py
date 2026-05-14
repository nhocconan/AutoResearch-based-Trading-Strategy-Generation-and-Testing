#!/usr/bin/env python3
# 4h_RSI_Trend_Filter_Volume
# Hypothesis: RSI(14) > 55 with price above 4h EMA20 and volume > 1.5x 20-period average for long; RSI < 45 with price below EMA20 and volume confirmation for short. Uses 12h EMA50 as trend filter to avoid counter-trend trades. Designed for 20-35 trades/year with clear trend and volume to avoid false signals. Works in bull via trend continuation and bear via reversals at extremes.

name = "4h_RSI_Trend_Filter_Volume"
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
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA20 for price filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        rsi_val = rsi[i]
        ema20_val = ema20[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # LONG: RSI > 55, price above EMA20, above 12h EMA50 trend, with volume confirmation
            if rsi_val > 55 and close[i] > ema20_val and close[i] > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 45, price below EMA20, below 12h EMA50 trend, with volume confirmation
            elif rsi_val < 45 and close[i] < ema20_val and close[i] < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI drops below 50 or price breaks below EMA20
            if rsi_val < 50 or close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI rises above 50 or price breaks above EMA20
            if rsi_val > 50 or close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals