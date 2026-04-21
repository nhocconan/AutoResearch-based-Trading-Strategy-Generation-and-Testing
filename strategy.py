#!/usr/bin/env python3
"""
12h_RSI_Pullback_1dTrend_Volume
Hypothesis: Use RSI(14) pullbacks to 40-60 range on 12h with 1d EMA50 trend filter and volume confirmation. Captures mean reversion within strong trends, works in bull/bear by following higher timeframe trend. Target 15-35 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === RSI(14) on 12h ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_1d = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: RSI pullback to 40-50 in uptrend + volume spike
            if (40 <= rsi_val <= 50 and 
                price_close > trend_1d and 
                vol_spike > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: RSI pullback to 50-60 in downtrend + volume spike
            elif (50 <= rsi_val <= 60 and 
                  price_close < trend_1d and 
                  vol_spike > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI reaches opposite extreme (60 for long, 40 for short)
            if position == 1 and rsi_val >= 60:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val <= 40:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_RSI_Pullback_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0