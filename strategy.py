#!/usr/bin/env python3
"""
Hypothesis: 6H RSI mean-reversion with weekly trend filter works because weekly RSI > 50 indicates
bullish regime where RSI dips are bought, while weekly RSI < 50 indicates bearish regime where
RSI bounces are sold. This adapts to both bull and bear markets by using weekly RSI as regime filter.
Entry: RSI(14) < 30 for long, > 70 for short on 6H chart with weekly RSI confirmation.
Exit: RSI returns to 50 level. Uses tight stop via signal reversal to limit drawdown.
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
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI(14) for regime
    delta = pd.Series(df_1w['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 6H timeframe (waits for weekly bar to close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # Calculate 6H RSI(14) for entry signals
    delta_6h = pd.Series(close).diff().values
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_6h = avg_gain_6h / (avg_loss_6h + 1e-10)
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup for RSI
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_6h[i]):
            signals[i] = 0.0
            continue
        
        rsi_6h_val = rsi_6h[i]
        weekly_rsi = rsi_1w_aligned[i]
        
        if position == 0:
            # Long: RSI oversold in bullish weekly regime (weekly RSI > 50)
            if rsi_6h_val < 30 and weekly_rsi > 50:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought in bearish weekly regime (weekly RSI < 50)
            elif rsi_6h_val > 70 and weekly_rsi < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or weekly regime turns bearish
            if rsi_6h_val >= 50 or weekly_rsi < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or weekly regime turns bullish
            if rsi_6h_val <= 50 or weekly_rsi > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI14_WeeklyRegime_MeanReversion"
timeframe = "6h"
leverage = 1.0