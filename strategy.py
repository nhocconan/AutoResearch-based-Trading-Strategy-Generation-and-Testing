#!/usr/bin/env python3
"""
1d_RSI_Reversal_Kelly_WeeklyTrend
Hypothesis: On daily timeframe, take mean-reversion entries when RSI(14) < 30 (oversold) in weekly uptrend or RSI(14) > 70 (overbought) in weekly downtrend. Position size dynamically scaled by Kelly criterion based on recent win rate and average win/loss ratio. Uses Kelly fraction capped at 0.30 to manage risk. Works in both bull and bear markets by aligning with weekly trend direction.
"""

name = "1d_RSI_Reversal_Kelly_WeeklyTrend"
timeframe = "1d"
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
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === DAILY RSI FOR MEAN REVERSION ===
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate win rate and win/loss ratio for Kelly (using last 60 days)
    lookback = 60
    returns = np.diff(close, prepend=close[0]) / close
    wins = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(lookback, n):
        start = i - lookback
        period_returns = returns[start:i+1]
        winning = period_returns[period_returns > 0]
        losing = period_returns[period_returns < 0]
        win_rate = len(winning) / lookback if lookback > 0 else 0.5
        avg_win = np.mean(winning) if len(winning) > 0 else 0.0
        avg_loss = np.mean(np.abs(losing)) if len(losing) > 0 else 0.01
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly fraction: f = (bp - q) / b where b = win_loss_ratio, p = win_rate, q = 1-p
        if win_loss_ratio > 0:
            kelly = (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio
            kelly = max(0, min(kelly, 0.30))  # Cap at 0.30, no negative
        else:
            kelly = 0.0
        
        wins[i] = win_rate
        losses[i] = win_loss_ratio
    
    # Forward fill for periods before lookback
    for i in range(1, lookback):
        wins[i] = wins[i-1] if i > 0 else 0.5
        losses[i] = losses[i-1] if i > 0 else 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(ema_21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kelly_size = wins[i] * losses[i]  # Simplified Kelly proxy using precomputed values
        if kelly_size > 0.30:
            kelly_size = 0.30
        elif kelly_size < 0.05:
            kelly_size = 0.0  # Minimum size threshold
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND weekly uptrend (close > weekly EMA21)
            if rsi[i] < 30 and close[i] > ema_21_1w_aligned[i]:
                signals[i] = kelly_size if kelly_size > 0 else 0.25
                position = 1
            # Short: RSI > 70 (overbought) AND weekly downtrend (close < weekly EMA21)
            elif rsi[i] > 70 and close[i] < ema_21_1w_aligned[i]:
                signals[i] = -kelly_size if kelly_size > 0 else -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI > 60 (overbought territory) OR price crosses below weekly EMA21
            if rsi[i] > 60 or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = wins[i] * losses[i]  # maintain position scaled by Kelly
                if signals[i] > 0.30:
                    signals[i] = 0.30
        elif position == -1:
            # Short exit: RSI < 40 (oversold territory) OR price crosses above weekly EMA21
            if rsi[i] < 40 or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -wins[i] * losses[i]  # maintain position scaled by Kelly
                if signals[i] < -0.30:
                    signals[i] = -0.30
    
    return signals