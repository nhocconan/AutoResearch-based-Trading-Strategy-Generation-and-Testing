#!/usr/bin/env python3
# 6h_12h_Momentum_Shift
# Hypothesis: Captures momentum shifts by combining 12h price momentum (ROC12) with 6h price action confirmation.
# Long when 12h ROC12 > 0 (bullish momentum) and 6h closes above 6h EMA34; short when 12h ROC12 < 0 (bearish momentum) and 6h closes below 6h EMA34.
# Uses volume confirmation to avoid false signals and reduce whipsaws.
# Designed for low trade frequency (target: 50-150 trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe momentum.

name = "6h_12h_Momentum_Shift"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for momentum calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h ROC12 for momentum ---
    close_12h = df_12h['close'].values
    roc_12 = np.zeros_like(close_12h)
    roc_12[12:] = (close_12h[12:] - close_12h[:-12]) / close_12h[:-12]
    roc_12_smooth = pd.Series(roc_12).ewm(span=3, adjust=False, min_periods=1).mean().values
    roc_12_aligned = align_htf_to_ltf(prices, df_12h, roc_12_smooth)
    
    # --- 6h EMA34 for entry timing ---
    ema_34_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ROC12 (12) and EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(roc_12_aligned[i]) or
            np.isnan(ema_34_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum direction from 12h ROC12
        bullish_momentum = roc_12_aligned[i] > 0
        bearish_momentum = roc_12_aligned[i] < 0
        
        if position == 0:
            if bullish_momentum and vol_surge[i]:
                # Long: 12h bullish momentum + volume surge + price above 6h EMA34
                if close[i] > ema_34_6h[i]:
                    signals[i] = 0.25
                    position = 1
            elif bearish_momentum and vol_surge[i]:
                # Short: 12h bearish momentum + volume surge + price below 6h EMA34
                if close[i] < ema_34_6h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 12h momentum turns bearish OR price crosses below EMA34
                if bearish_momentum or close[i] < ema_34_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 12h momentum turns bullish OR price crosses above EMA34
                if bullish_momentum or close[i] > ema_34_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals