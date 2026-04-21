#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI2_Confirm_v3
Hypothesis: Use KAMA direction on 12h as primary trend filter, with RSI(2) for entry timing on 12h, and volume confirmation. Designed to capture trend continuation moves with mean-reversion entries in the direction of the trend. Uses 1d timeframe for trend confirmation to avoid false signals. Target 12-30 trades/year on 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h and 1d HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h KAMA trend (ER=10) ===
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.abs(np.diff(close_12h, 1))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # === 1d trend: EMA34 for confirmation ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h RSI(2) for entry timing ===
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2_aligned = align_htf_to_ltf(prices, df_12h, rsi_2)
    
    # === Volume confirmation: 20-period volume average on 12h ===
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume_12h / vol_ma_20, 1.0)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama_12h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(rsi_2_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_12h_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        rsi_val = rsi_2_aligned[i]
        vol_spike = vol_ratio_aligned[i]
        
        if position == 0:
            # Long: Price above both KAMA and 1d EMA34, RSI(2) < 15 (oversold), volume spike
            if (price_close > kama_val and 
                price_close > ema_1d and 
                rsi_val < 15 and 
                vol_spike > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Price below both KAMA and 1d EMA34, RSI(2) > 85 (overbought), volume spike
            elif (price_close < kama_val and 
                  price_close < ema_1d and 
                  rsi_val > 85 and 
                  vol_spike > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses KAMA in opposite direction
            if position == 1 and price_close < kama_val:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_Trend_RSI2_Confirm_v3"
timeframe = "12h"
leverage = 1.0