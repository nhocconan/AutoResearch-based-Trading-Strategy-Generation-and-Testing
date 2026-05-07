#!/usr/bin/env python3
"""
1H_Volume_Weighted_RSI_Momentum_4H_Trend_Filter
Hypothesis: Use 4h trend direction (EMA50) for bias, 1h RSI for momentum, and volume-weighted RSI for entry timing. Volume filter ensures participation. Works in bull/bear by following 4h trend. Targets 15-37 trades/year on 1h via strict 4h trend + volume + RSI confluence.
"""
name = "1H_Volume_Weighted_RSI_Momentum_4H_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([np.array([0]), gain])
    loss = np.concatenate([np.array([0]), loss])
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume-weighted RSI (VW-RSI)
    vw_close = close * volume
    vw_delta = np.diff(vw_close)
    vw_gain = np.where(vw_delta > 0, vw_delta, 0)
    vw_loss = np.where(vw_delta < 0, -vw_delta, 0)
    vw_gain = np.concatenate([np.array([0]), vw_gain])
    vw_loss = np.concatenate([np.array([0]), vw_loss])
    
    avg_vw_gain = pd.Series(vw_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vw_loss = pd.Series(vw_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vw_rs = np.where(avg_vw_loss != 0, avg_vw_gain / avg_vw_loss, 0)
    vw_rsi = 100 - (100 / (1 + vw_rs))
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vw_rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend, RSI > 50, VW-RSI > RSI (bullish volume), volume confirmation
            if (close[i] > ema_50_4h_aligned[i] and 
                rsi[i] > 50 and 
                vw_rsi[i] > rsi[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, RSI < 50, VW-RSI < RSI (bearish volume), volume confirmation
            elif (close[i] < ema_50_4h_aligned[i] and 
                  rsi[i] < 50 and 
                  vw_rsi[i] < rsi[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h downtrend or RSI < 40
            if (close[i] < ema_50_4h_aligned[i] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h uptrend or RSI > 60
            if (close[i] > ema_50_4h_aligned[i] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals