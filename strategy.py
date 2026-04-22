#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Band width regime filter with 1-week RSI trend filter.
- Long when BB width > 60th percentile (volatile/trending) and weekly RSI > 50 (bullish bias)
- Short when BB width > 60th percentile and weekly RSI < 50 (bearish bias)
- Avoid trades when BB width < 40th percentile (low volatility/chop)
Uses volatility expansion to capture trending moves while avoiding choppy periods.
Weekly RSI provides multi-timeframe trend bias to avoid counter-trend trades.
Works in both bull and bear markets by following volatility expansion in the direction of higher timeframe trend.
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
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std_dev
    lower = sma - bb_std * std_dev
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Weekly RSI for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # BB width percentile rank (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(bb_width_rank[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry: BB width in expansion regime (>60th percentile) + weekly RSI bias
            if bb_width_rank[i] > 0.6:
                if rsi_1w_aligned[i] > 50:  # Bullish bias
                    signals[i] = 0.25
                    position = 1
                elif rsi_1w_aligned[i] < 50:  # Bearish bias
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: BB width contraction (<40th percentile) or RSI crosses 50
            exit_signal = False
            
            if bb_width_rank[i] < 0.4:  # Low volatility/chop
                exit_signal = True
            elif position == 1 and rsi_1w_aligned[i] < 50:  # Long but turned bearish
                exit_signal = True
            elif position == -1 and rsi_1w_aligned[i] > 50:  # Short but turned bullish
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BBWidth_Regime_1wRSI_Trend"
timeframe = "6h"
leverage = 1.0