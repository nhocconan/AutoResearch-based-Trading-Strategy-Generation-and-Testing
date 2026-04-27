#!/usr/bin/env python3
"""
1d_RSI14_MeanReversion_WeeklyTrend
Hypothesis: RSI mean reversion works in ranging markets while weekly trend filter
avoids counter-trend trades in strong trends. Weekly trend provides structural bias
to improve win rate in both bull and bear markets. Targets 15-25 trades/year
on 1d timeframe to minimize fee drag.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and weekly EMA
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if weekly trend not ready
        if np.isnan(ema20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_values[i]
        weekly_trend = ema20_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold in uptrend or neutral trend
            if rsi_val < 30 and price > weekly_trend:
                signals[i] = size
                position = 1
            # Short: RSI overbought in downtrend or neutral trend
            elif rsi_val > 70 and price < weekly_trend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend turns down
            if rsi_val > 70 or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or trend turns up
            if rsi_val < 30 or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI14_MeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0