#!/usr/bin/env python3
"""
1d_1W_SimpleTrend_Momentum_v1
Hypothesis: Use weekly EMA34 for trend direction and daily RSI(14) for momentum entry, filtered by volume > 1.5x 20-day average. Only trade in the direction of the weekly trend. Weekly trend provides robustness across bull/bear cycles, while daily RSI captures short-term momentum. Volume confirmation ensures institutional participation. Designed for low trade frequency (target 10-25 trades/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not available
        if np.isnan(ema34_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA
        weekly_uptrend = close_1w[-1] > ema34_1w[-1] if len(close_1w) > 0 else False  # Simplified: use current weekly close vs EMA
        # Better: use aligned weekly EMA vs price (but price is daily, so approximate)
        # Instead: use slope of weekly EMA
        if i >= 35:
            weekly_uptrend = ema34_1w_aligned[i] > ema34_1w_aligned[i-1]
        else:
            weekly_uptrend = False
        
        # Only take longs in uptrend, shorts in downtrend
        if position == 0:
            # Long: weekly uptrend + RSI > 50 (bullish momentum) + volume confirmation
            if weekly_uptrend and rsi[i] > 50 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI < 50 (bearish momentum) + volume confirmation
            elif not weekly_uptrend and rsi[i] < 50 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI < 40 (loss of momentum)
            if not weekly_uptrend or rsi[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI > 60 (loss of bearish momentum)
            if weekly_uptrend or rsi[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_SimpleTrend_Momentum_v1"
timeframe = "1d"
leverage = 1.0