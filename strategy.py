#!/usr/bin/env python3
# 1H_4H_1D_Triple_Momentum_Confluence
# Hypothesis: Price momentum aligned across 1h, 4h, and 1d timeframes indicates strong trend continuation.
# Long when: 1h RSI > 55, 4h close > 4h EMA20, 1d close > 1d EMA50.
# Short when: 1h RSI < 45, 4h close < 4h EMA20, 1d close < 1d EMA50.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods.
# Position size: 0.20 (20% of capital) to control drawdown.
# Target: 20-40 trades/year per symbol to minimize fee drag while capturing major moves.

name = "1H_4H_1D_Triple_Momentum_Confluence"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA20
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Long conditions: bullish momentum across all timeframes
        long_condition = (rsi[i] > 55) and (close[i] > ema20_4h_aligned[i]) and (close[i] > ema50_1d_aligned[i])
        
        # Short conditions: bearish momentum across all timeframes
        short_condition = (rsi[i] < 45) and (close[i] < ema20_4h_aligned[i]) and (close[i] < ema50_1d_aligned[i])
        
        if position == 0:
            # Enter long
            if long_condition:
                signals[i] = 0.20
                position = 1
            # Enter short
            elif short_condition:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: any momentum breaks down
            if not long_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: any momentum breaks down
            if not short_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals