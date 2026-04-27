#!/usr/bin/env python3
"""
1d_RSI_MeanReversion_WeeklyTrend_Filter
Hypothesis: Uses weekly trend filter with daily RSI mean reversion. In strong weekly uptrend, buy RSI<30; in strong weekly downtrend, sell RSI>70. Weekly trend avoids counter-trend trades in major moves, while RSI captures pullbacks. Designed for low frequency (10-25 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly EMA50 to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 50  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema50 = ema50_1w_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Determine weekly trend
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and rsi_val < 30:
                # Long in uptrend on RSI oversold
                signals[i] = size
                position = 1
            elif downtrend and rsi_val > 70:
                # Short in downtrend on RSI overbought
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI reaches neutral (50) or trend change
            if rsi_val >= 50 or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI reaches neutral (50) or trend change
            if rsi_val <= 50 or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_MeanReversion_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0