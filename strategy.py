#!/usr/bin/env python3
"""
1d_1w_Kelly_Fractional_Kelly_Strategy
Hypothesis: Use 1D price action with 1W trend filter and Kelly criterion position sizing.
Long when price > 1W EMA50 and RSI(14) < 40 (oversold in uptrend).
Short when price < 1W EMA50 and RSI(14) > 60 (overbought in downtrend).
Position size scaled by Kelly fraction: f = (bp - q)/b where b=win/loss ratio, p=win probability.
Uses 10-period win/loss tracking to estimate p and b. Max position 0.30.
Designed for low trade frequency (<25/year) with asymmetric risk control.
Works in bull via trend-following oversold bounces, in bear via trend-following overbought fades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1W data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1W
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * 2/51) + (ema50_1w[i-1] * (1 - 2/51))
    
    # Align 1W EMA50 to 1D (wait for weekly close)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate RSI(14) on 1D
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Track trade performance for Kelly sizing
    win_count = 0
    loss_count = 0
    total_return = 0.0
    last_position = 0
    entry_price = 0.0
    
    signals = np.zeros(n)
    
    start_idx = max(50, 14)  # need EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Kelly fraction from historical performance
        if win_count + loss_count >= 10:
            win_prob = win_count / (win_count + loss_count)
            if loss_count > 0:
                avg_win = total_return / win_count if win_count > 0 else 0
                avg_loss = abs(total_return) / loss_count if loss_count > 0 else 0
                if avg_loss > 0:
                    b = avg_win / avg_loss  # win/loss ratio
                    kelly = (b * win_prob - (1 - win_prob)) / b if b > 0 else 0
                    kelly = max(0, min(kelly, 0.30))  # cap at 30%, no negative
                else:
                    kelly = 0.30
            else:
                kelly = 0.30
        else:
            kelly = 0.15  # reduced size until sufficient data
        
        # Determine signal based on trend and RSI extremes
        if ema50_1w_aligned[i] > 0:  # trend filter available
            if close[i] > ema50_1w_aligned[i] and rsi[i] < 40:
                # Oversold in uptrend - long
                target_size = kelly
            elif close[i] < ema50_1w_aligned[i] and rsi[i] > 60:
                # Overbought in downtrend - short
                target_size = -kelly
            else:
                target_size = 0.0
        else:
            target_size = 0.0
        
        # Change signal only if different from previous (reduce churn)
        if i == start_idx:
            signals[i] = target_size
        else:
            # Only change if significant difference (>0.05) to reduce churn
            if abs(target_size - signals[i-1]) > 0.05:
                signals[i] = target_size
            else:
                signals[i] = signals[i-1]
    
    return signals

name = "1d_1w_Kelly_Fractional_Kelly_Strategy"
timeframe = "1d"
leverage = 1.0