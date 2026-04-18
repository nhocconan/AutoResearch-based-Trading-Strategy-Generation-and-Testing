#!/usr/bin/env python3
"""
1d_RSI20_Bounce_1wTrend_Filter_v1
Hypothesis: On daily timeframe, buy when RSI(14) < 20 (deep oversold) and weekly trend is up (price > weekly EMA50); sell when RSI > 80 (overbought) and weekly trend is down (price < weekly EMA50). Uses weekly trend filter to avoid counter-trend trades in strong trends, reducing false signals. Designed for low trade frequency (10-20/year) with high win rate in both bull and bear markets by catching mean-reversion bounces in alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for EMA50 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # ensure enough data for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_1w_aligned[i]
        rsi_val = rsi_values[i]
        
        if position == 0:
            # Long: RSI < 20 (oversold) and weekly uptrend (price > weekly EMA50)
            if rsi_val < 20 and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 80 (overbought) and weekly downtrend (price < weekly EMA50)
            elif rsi_val > 80 and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI > 60 (exit overbought) or trend breaks down
            if rsi_val > 60 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI < 40 (exit oversold) or trend breaks up
            if rsi_val < 40 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_RSI20_Bounce_1wTrend_Filter_v1"
timeframe = "1d"
leverage = 1.0