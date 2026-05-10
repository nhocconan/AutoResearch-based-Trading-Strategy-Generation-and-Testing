#!/usr/bin/env python3
# 6h_ConnorsRSI_14_2_100_SMA200_Filter
# Hypothesis: Connors RSI (CRSI) identifies overbought/oversold extremes with high mean reversion probability.
# Long when CRSI < 15 and price > 200-day SMA (uptrend filter). Short when CRSI > 85 and price < 200-day SMA (downtrend filter).
# Uses 1-day timeframe for trend filter to avoid whipsaws. Designed for low trade frequency (~25-35/year) to minimize fee drift.
# Works in bull/bear markets: trend filter ensures we only trade with the higher timeframe trend, CRSI captures pullbacks.

name = "6h_ConnorsRSI_14_2_100_SMA200_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period):
    """Relative Strength Index with Wilder's smoothing."""
    close = pd.Series(close)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = pd.Series(up).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = pd.Series(down).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def connors_rsi(close, rsi_len=3, streak_len=2, rank_len=100):
    """Connors RSI = (RSI(3) + RSI_streak(2) + PercentRank(100)) / 3"""
    # RSI(3)
    rsi_val = rsi(close, rsi_len)
    
    # Streak RSI: RSI of up/down streak lengths
    streak = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    streak = np.abs(streak)  # streak length regardless of direction
    streak_rsi = rsi(streak, streak_len)
    
    # Percent Rank of current close over lookback period
    rank = np.full_like(close, np.nan, dtype=float)
    for i in range(rank_len-1, len(close)):
        window = close[i-rank_len+1:i+1]
        rank[i] = (np.sum(window < close[i]) + 0.5 * np.sum(window == close[i])) / rank_len * 100
    
    crsi = (rsi_val + streak_rsi + rank) / 3
    return crsi

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Connors RSI components
    crsi = connors_rsi(close, 3, 2, 100)
    
    # Get 1-day data for trend filter (200-day SMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough history for SMA200
    
    for i in range(start_idx, n):
        if np.isnan(crsi[i]) or np.isnan(sma200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_above_sma200 = close[i] > sma200_1d_aligned[i]
        price_below_sma200 = close[i] < sma200_1d_aligned[i]
        
        if position == 0:
            # Long: CRSI oversold (<15) and price above 200-day SMA (uptrend)
            if crsi[i] < 15 and price_above_sma200:
                signals[i] = 0.25
                position = 1
            # Short: CRSI overbought (>85) and price below 200-day SMA (downtrend)
            elif crsi[i] > 85 and price_below_sma200:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CRSI > 50 (mean reversion complete) or trend breaks
            if crsi[i] > 50 or not price_above_sma200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CRSI < 50 (mean reversion complete) or trend breaks
            if crsi[i] < 50 or not price_below_sma200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals