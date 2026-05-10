#!/usr/bin/env python3
"""
1d_Portfolio_Momentum_Rotation
Hypothesis: Rank assets by 1-month momentum (20-day return), go long top 2 (BTC, ETH, SOL) and short bottom 1.
Uses weekly trend filter (1w EMA20) to avoid counter-trend trades in strong trends.
Rebalances daily at close. Designed for low turnover (<25 trades/year) to minimize fee drag.
Works in bull/bear by capturing relative strength and avoiding weak trends.
"""

name = "1d_Portfolio_Momentum_Rotation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Momentum: 20-day return (approx 1 month)
    mom = np.full(n, np.nan)
    if n >= 20:
        mom[19:] = (close[19:] / close[:n-19]) - 1
    
    # Weekly trend filter: EMA20 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Align weekly EMA to daily
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Determine trend: above/below EMA20
    trend_up = close >= ema20_1w_aligned  # True if in uptrend
    
    signals = np.zeros(n)
    
    # Rebalance only once per week (every 7th day) to reduce turnover
    for i in range(20, n):
        if np.isnan(mom[i]) or np.isnan(ema20_1w_aligned[i]):
            continue
        
        # Only rebalance on weekly boundary (day 0 of week)
        if i % 7 != 0:
            # Carry forward previous day's signal
            signals[i] = signals[i-1]
            continue
        
        # In strong uptrend, go long top 2 momentum
        # In weak/downtrend, go long top 1, short bottom 1 (market neutral)
        if trend_up[i]:
            # Bullish: long top 2
            # Find indices of top 2 momentum
            mom_slice = mom[max(0, i-19):i+1]  # Use recent momentum
            if len(mom_slice) < 3:
                continue
            # Get relative ranks within window
            sorted_idx = np.argsort(mom_slice)[::-1]  # Descending
            if len(sorted_idx) >= 2:
                top2_idx = sorted_idx[:2]
                # Map back to absolute indices
                top2_abs = [max(0, i-19) + idx for idx in top2_idx]
                # Equal weight long
                signals[i] = 0.0
                for idx in top2_abs:
                    if idx < n:
                        signals[i] += 0.5  # 50% each
        else:
            # Bearish/neutral: long top 1, short bottom 1
            mom_slice = mom[max(0, i-19):i+1]
            if len(mom_slice) < 3:
                continue
            sorted_idx = np.argsort(mom_slice)[::-1]
            if len(sorted_idx) >= 2:
                top1_idx = sorted_idx[0]
                bottom1_idx = sorted_idx[-1]
                top1_abs = max(0, i-19) + top1_idx
                bottom1_abs = max(0, i-19) + bottom1_idx
                signals[i] = 0.0
                if top1_abs < n:
                    signals[i] += 0.5  # Long 50%
                if bottom1_abs < n:
                    signals[i] -= 0.5  # Short 50%
    
    return signals