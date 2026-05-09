#!/usr/bin/env python3
# 6H_1D_1W_RelativeStrength_Carry_Trend
# Hypothesis: Use 1d relative strength (RSI) to identify leading/lagging assets and 1w trend filter.
# Long when BTC/ETH shows relative strength (RSI > 50) and 1w uptrend, short when weak (RSI < 50) and 1w downtrend.
# Enter on 6d pullbacks to EMA(21) for better risk/reward. Target: 15-25 trades/year per symbol.

name = "6H_1D_1W_RelativeStrength_Carry_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(20) for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema_20_1w
    
    # Align 1d RSI and 1w trend to 6h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # 6h EMA(21) for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi_aligned[i]) or np.isnan(trend_up_1w_aligned[i]) or np.isnan(ema_21[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI > 50 (relative strength) + 1w uptrend + pullback to EMA(21)
            if rsi_aligned[i] > 50 and trend_up_1w_aligned[i] and close[i] <= ema_21[i] * 1.005:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI < 50 (relative weakness) + 1w downtrend + pullback to EMA(21)
            elif rsi_aligned[i] < 50 and not trend_up_1w_aligned[i] and close[i] >= ema_21[i] * 0.995:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI < 40 (loss of strength) or 1w trend turns down
            if rsi_aligned[i] < 40 or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 60 (regained strength) or 1w trend turns up
            if rsi_aligned[i] > 60 or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals