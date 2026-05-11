# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_RSI_Extreme_With_Volume_Confirmation
Hypothesis: Trade extreme RSI(14) readings (≤10 for long, ≥90 for short) with volume confirmation and 12h EMA50 trend filter. RSI extremes occur in pullbacks during trends, providing high-probability entries. Volume confirms institutional interest. Trend filter ensures alignment with higher timeframe momentum. Designed for low trade frequency (<30/year) to minimize fee drag while capturing mean-reversion within trends. Works in bull/bear markets via trend filter.
"""

name = "4h_RSI_Extreme_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h EMA50 Trend Filter ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- RSI Calculation ---
    rsi = calculate_rsi(close, 14)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: RSI ≤ 10 (extreme oversold) + above 12h EMA50 + volume spike
            if (rsi[i] <= 10 and 
                close[i] > ema_50_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: RSI ≥ 90 (extreme overbought) + below 12h EMA50 + volume spike
            elif (rsi[i] >= 90 and 
                  close[i] < ema_50_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI ≥ 50 (mean reversion complete) OR trend turns down
                if rsi[i] >= 50 or close[i] < ema_50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI ≤ 50 (mean reversion complete) OR trend turns up
                if rsi[i] <= 50 or close[i] > ema_50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals