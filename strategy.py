#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: KAMA trend (ER=10, slow=2) ===
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    vol = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder
    # Correct ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            direction = np.abs(close_1d[i] - close_1d[i-10])
            volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_trend = kama > np.roll(kama, 1)  # rising trend
    
    # === 1d: RSI(14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = gain_ma / np.where(loss_ma > 0, loss_ma, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Get values
        kama_trend_val = kama_trend_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_trend_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA uptrend + RSI > 50 + volume confirmation
            if kama_trend_val and rsi_val > 50 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI < 50 + volume confirmation
            elif not kama_trend_val and rsi_val < 50 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA downtrend OR RSI < 40
            if not kama_trend_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA uptrend OR RSI > 60
            if kama_trend_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals