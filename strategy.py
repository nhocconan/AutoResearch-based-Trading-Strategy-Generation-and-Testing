# 1h_1d_Momentum_Filter_Strategy
# Hypothesis: In 1h timeframe, use 1d momentum (price above/below 200 EMA) for directional bias,
# and 4h volatility contraction/expansion (BB width percentile) for entry timing.
# This avoids overtrading by requiring both trend alignment and volatility breakout.
# Works in bull/bear: long when price>200EMA + BB breakout up, short when price<200EMA + BB breakout down.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "1h_1d_Momentum_Filter_Strategy"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for 200 EMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on 1d close
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2/201) + (ema_200_1d[i-1] * (1 - 2/201))
    
    # Align 200 EMA to 1h
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 4h data for Bollinger Bands (volatility filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 4h close
    close_4h = df_4h['close'].values
    sma_20_4h = np.full_like(close_4h, np.nan)
    std_20_4h = np.full_like(close_4h, np.nan)
    
    for i in range(len(close_4h)):
        if i >= 19:
            sma_20_4h[i] = np.mean(close_4h[i-19:i+1])
            std_20_4h[i] = np.std(close_4h[i-19:i+1])
    
    upper_bb_4h = sma_20_4h + (2 * std_20_4h)
    lower_bb_4h = sma_20_4h - (2 * std_20_4h)
    bb_width_4h = upper_bb_4h - lower_bb_4h
    
    # Calculate BB width percentile (lookback 50 periods) to detect expansion
    bb_width_percentile = np.full_like(bb_width_4h, np.nan)
    for i in range(len(bb_width_4h)):
        if i >= 49:
            window = bb_width_4h[i-49:i+1]
            valid = ~np.isnan(window)
            if np.sum(valid) >= 10:  # minimum valid samples
                rank = np.sum(window[valid] <= bb_width_4h[i]) / np.sum(valid)
                bb_width_percentile[i] = rank * 100
    
    # Align BB width percentile to 1h
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_4h, bb_width_percentile)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(200, 50)  # 200 for EMA, 50 for BB percentile
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above 1d 200 EMA AND BB width expanding (breakout up)
        if close[i] > ema_200_1d_aligned[i] and bb_width_percentile_aligned[i] > 80:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short conditions: price below 1d 200 EMA AND BB width expanding (breakout down)
        elif close[i] < ema_200_1d_aligned[i] and bb_width_percentile_aligned[i] > 80:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: BB contraction (low volatility) or opposite momentum
        elif bb_width_percentile_aligned[i] < 20:  # volatility contraction
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
        # Hold current position
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals