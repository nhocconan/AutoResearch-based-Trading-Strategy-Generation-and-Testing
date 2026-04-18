#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_RSI_Pullback
Hypothesis: In both bull and bear markets, price often pulls back to key moving averages during trends. 
This strategy combines RSI momentum with adaptive Kelly position sizing based on recent win rate, 
using 60-period EMA as dynamic support/resistance. Designed for 6h timeframe to avoid overtrading.
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
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(gain)):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.mean(gain[:i+1])
                avg_loss[i] = np.mean(loss[:i+1])
            else:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA60 as dynamic trend filter
    ema60 = np.zeros_like(close)
    if len(close) >= 60:
        k = 2 / (60 + 1)
        ema60[59] = np.mean(close[:60])
        for i in range(60, len(close)):
            ema60[i] = close[i] * k + ema60[i-1] * (1 - k)
    
    # Win rate calculation for adaptive Kelly (lookback 50 trades)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / close[i-1]
    
    win_rate = np.full(n, 0.5)  # Start with 50%
    win_count = 0
    loss_count = 0
    min_lookback = 50
    
    for i in range(min_lookback, n):
        # Count wins/losses in lookback window
        window_returns = returns[i-min_lookback:i]
        wins = np.sum(window_returns > 0)
        losses = np.sum(window_returns < 0)
        total = wins + losses
        if total > 0:
            win_rate[i] = wins / total
        else:
            win_rate[i] = 0.5
    
    # Kelly fraction: f = (bp - q) / b where b=1 (1:1 payoff), p=win_rate, q=1-pwin_rate
    kelly_fraction = np.maximum(0, win_rate - (1 - win_rate))  # Simplified for b=1
    kelly_fraction = np.minimum(kelly_fraction, 0.5)  # Cap at 0.5 for safety
    position_size = kelly_fraction * 0.5  # Quarter Kelly for safety
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema60[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: RSI < 40 (pullback) and price above EMA60 (uptrend)
            if rsi[i] < 40 and close[i] > ema60[i]:
                signals[i] = position_size[i]
                position = 1
            # Short: RSI > 60 (pullback) and price below EMA60 (downtrend)
            elif rsi[i] > 60 and close[i] < ema60[i]:
                signals[i] = -position_size[i]
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or price below EMA60 (trend change)
            if rsi[i] > 60 or close[i] < ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or price above EMA60 (trend change)
            if rsi[i] < 40 or close[i] > ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals

name = "6h_Adaptive_Kelly_RSI_Pullback"
timeframe = "6h"
leverage = 1.0