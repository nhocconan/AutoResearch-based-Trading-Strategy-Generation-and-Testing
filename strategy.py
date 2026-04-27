#!/usr/bin/env python3
"""
1h_MultiTF_Trend_With_Volume_Filter
Hypothesis: Use 4h EMA trend and 1d RSI filter for direction, enter on 1h EMA cross with volume confirmation. 
Works in bull via 4h uptrend + 1h golden crosses, in bear via 4h downtrend + 1h death crosses.
Volume filter ensures momentum behind moves. Target 15-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend direction
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d RSI for overbought/oversold filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_vals = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_vals)
    
    # 1h EMA cross for entry timing (fast=12, slow=26)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_cross = ema_fast - ema_slow
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_cross[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        ema_4h_val = ema_4h_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        cross_val = ema_cross[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: 4h uptrend, 1h golden cross, not overbought, volume confirmation
            if (close[i] > ema_4h_val and 
                cross_val > 0 and 
                rsi_1d_val < 70 and 
                vol_conf):
                signals[i] = size
                position = 1
            # Short: 4h downtrend, 1h death cross, not oversold, volume confirmation
            elif (close[i] < ema_4h_val and 
                  cross_val < 0 and 
                  rsi_1d_val > 30 and 
                  vol_conf):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: 4h trend turns down OR 1h death cross
            if (close[i] < ema_4h_val or cross_val < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: 4h trend turns up OR 1h golden cross
            if (close[i] > ema_4h_val or cross_val > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_MultiTF_Trend_With_Volume_Filter"
timeframe = "1h"
leverage = 1.0