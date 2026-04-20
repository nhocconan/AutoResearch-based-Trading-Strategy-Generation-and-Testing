#!/usr/bin/env python3
"""
12h_1w_4h_Keltner_Breakout_Signal
Concept: Use 1-week Keltner channels to define trend and volatility, 4h for momentum confirmation, 12h for entry timing.
- Long: Close > Keltner Upper (1w) AND 4h RSI > 50 AND 12h volume > 1.5x 24-period average
- Short: Close < Keltner Lower (1w) AND 4h RSI < 50 AND 12h volume > 1.5x 24-period average
- Exit: Close crosses back below/above 20-period EMA (12h)
- Position sizing: 0.28
- Designed for low frequency (<30 trades/year) and robustness in bull/bear markets via volatility-adaptive channels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_4h_Keltner_Breakout_Signal"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # === 1w: Keltner Channel (20, 1.5) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # EMA20 of typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    ema_tp = pd.Series(tp_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # ATR(20)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = high_1w[0] - close_1w[0]
    tr3[0] = low_1w[0] - close_1w[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    keltner_upper = ema_tp + 1.5 * atr
    keltner_lower = ema_tp - 1.5 * atr
    
    # Align Keltner bands to 12h
    keltner_upper_12h = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_12h = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # === 4h: RSI(14) ===
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_4h = align_htf_to_ltf(prices, df_4h, rsi)
    
    # === 12h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA20 for exit
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume: 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Get values
        kup = keltner_upper_12h[i]
        klow = keltner_lower_12h[i]
        rsi_val = rsi_4h[i]
        vol_ma_val = vol_ma[i]
        vol = volume[i]
        close_val = close[i]
        ema_val = ema_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(kup) or np.isnan(klow) or np.isnan(rsi_val) or 
            np.isnan(vol_ma_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 24-period average
        vol_condition = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: close above Keltner Upper AND RSI > 50 AND volume confirmation
            if close_val > kup and rsi_val > 50 and vol_condition:
                signals[i] = 0.28
                position = 1
            # Short: close below Keltner Lower AND RSI < 50 AND volume confirmation
            elif close_val < klow and rsi_val < 50 and vol_condition:
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Long exit: close below EMA20
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Short exit: close above EMA20
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals