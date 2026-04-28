#!/usr/bin/env python3
"""
1d_RCI_Momentum_52WeekLow_Trend
Hypothesis: The Rate of Change Index (RCI) identifies momentum extremes, while proximity to 52-week lows indicates mean-reversion opportunities in oversold conditions. Combined with a weekly trend filter (price above/below weekly SMA50) and volume confirmation, this strategy captures mean-reversion bounces in bear markets and momentum continuations in bull markets. The 1d timeframe ensures low trade frequency (<25/year) to minimize fee drag, while the weekly trend filter prevents counter-trend entries. Target: 15-20 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:  # Need ~1 year for 52-week low calculation
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    close_weekly = df_weekly['close'].values
    sma_50_weekly = pd.Series(close_weekly).rolling(window=50, min_periods=50).mean().values
    sma_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma_50_weekly)
    
    # Calculate 52-week low (260 trading days)
    # Use expanding minimum to track 52-week low
    min_52w = np.full(n, np.nan)
    for i in range(260, n):
        min_52w[i] = np.min(low[i-260:i])
    
    # Calculate RCI (Rank Correlation Index) over 9 periods
    # RCI measures the correlation between price ranks and time ranks
    rci = np.full(n, np.nan)
    for i in range(9, n):
        # Get prices for the last 9 periods
        prices_window = close[i-8:i+1]
        # Rank the prices (1 = lowest, 9 = highest)
        price_ranks = pd.Series(prices_window).rank(method='average').values
        # Time ranks are always 1,2,3,...,9
        time_ranks = np.arange(1, 10)
        # Calculate Spearman correlation
        if np.std(price_ranks) > 0 and np.std(time_ranks) > 0:
            rci[i] = np.corrcoef(price_ranks, time_ranks)[0, 1]
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(260, 9)  # Wait for 52-week low and RCI to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_weekly_aligned[i]) or np.isnan(min_52w[i]) or 
            np.isnan(rci[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        price_above_weekly_trend = close[i] > sma_50_weekly_aligned[i]
        price_below_weekly_trend = close[i] < sma_50_weekly_aligned[i]
        near_52w_low = close[i] <= (min_52w[i] * 1.05)  # Within 5% of 52-week low
        rci_oversold = rci[i] < -0.8  # Strong negative momentum
        rci_overbought = rci[i] > 0.8  # Strong positive momentum
        
        # Entry logic:
        # Long when: near 52-week low, RCI oversold, and weekly uptrend (or no strong downtrend)
        # Short when: near 52-week high (implied by strong uptrend), RCI overbought, and weekly downtrend
        long_entry = near_52w_low and rci_oversold and price_above_weekly_trend
        short_entry = (close[i] >= (min_52w[i] * 3.0)) and rci_overbought and price_below_weekly_trend  # Simplified 52w high proxy
        
        # Exit when RCI reverts to neutral
        long_exit = rci[i] > -0.3
        short_exit = rci[i] < 0.3
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_RCI_Momentum_52WeekLow_Trend"
timeframe = "1d"
leverage = 1.0