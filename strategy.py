#!/usr/bin/env python3
"""
Hypothesis: 4h price above/below 1d 200 EMA + 4h RSI(14) extreme + 1d volume confirmation.
- Long when: price > EMA200_1d AND RSI(14) < 30 AND 1d volume > 1.5x 20-day average
- Short when: price < EMA200_1d AND RSI(14) > 70 AND 1d volume > 1.5x 20-day average
- EMA200 acts as dynamic support/resistance; RSI extremes signal exhaustion/reversal
- Volume confirms institutional participation in the move
- Works in bull/bear by using EMA200 as trend filter (avoids counter-trend trades)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === 1d EMA200 ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_current = df_1d['volume'].values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 1d volume > 1.5x 20-day average
        vol_spike = vol_1d_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Price relative to EMA200
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA200 + RSI oversold + volume spike
            if price_above_ema and rsi_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA200 + RSI overbought + volume spike
            elif price_below_ema and rsi_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite condition or RSI normalization
        elif position == 1:
            # Exit long if price breaks below EMA200 or RSI > 50
            if not price_above_ema or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above EMA200 or RSI < 50
            if not price_below_ema or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA200_RSI14_Volume1.5x"
timeframe = "4h"
leverage = 1.0