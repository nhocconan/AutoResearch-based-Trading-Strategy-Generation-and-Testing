#!/usr/bin/env python3
"""
1d_Keltner_Squeeze_RSI_v1
Keltner squeeze (BBW/KCW < 0.7) + RSI(14) > 60 for long, < 40 for short.
Uses 1w EMA50 as trend filter: long only when price > 1w EMA50, short only when price < 1w EMA50.
Exit when squeeze releases (BBW/KCW >= 1.0) or RSI crosses midline (40-60).
Designed to capture volatility expansion moves with momentum confirmation in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year).
"""

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
    close_series = pd.Series(close)
    
    # === Bollinger Bands (20, 2) ===
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # === Keltner Channel (20, 1.5) ===
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    kc_mid = tp_series.rolling(window=20, min_periods=20).mean().values
    tp_ma = tp_series.rolling(window=20, min_periods=20).mean()
    tp_dev = (tp_series - tp_ma).abs()
    kc_dev = tp_dev.rolling(window=20, min_periods=20).mean().values
    kc_upper = kc_mid + 1.5 * kc_dev
    kc_lower = kc_mid - 1.5 * kc_dev
    kc_width = (kc_upper - kc_lower) / kc_mid
    
    # === Squeeze indicator (BBW / KCW) ===
    squeeze = bb_width / kc_width
    
    # === RSI(14) ===
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # === 1w EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: squeeze < 0.7, RSI > 60, price above 1w EMA50
            if (squeeze[i] < 0.7 and 
                rsi[i] > 60 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: squeeze < 0.7, RSI < 40, price below 1w EMA50
            elif (squeeze[i] < 0.7 and 
                  rsi[i] < 40 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: squeeze >= 1.0 OR RSI < 40
            if (squeeze[i] >= 1.0 or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: squeeze >= 1.0 OR RSI > 60
            if (squeeze[i] >= 1.0 or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Squeeze_RSI_v1"
timeframe = "1d"
leverage = 1.0