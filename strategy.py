#!/usr/bin/env python3
# 4h_RCI_WeeklyTrend_Volume
# Hypothesis: Weekly RCI (Rank Correlation Index) for long-term trend + 4h RCI overbought/oversold with volume confirmation.
# Long when: weekly RCI > 0 (uptrend) and 4h RCI < -80 (oversold) and volume > 1.5x 20-period average.
# Short when: weekly RCI < 0 (downtrend) and 4h RCI > 80 (overbought) and volume > 1.5x 20-period average.
# Exit when 4h RCI crosses back above -20 (for long) or below 20 (for short).
# Uses RCI to catch mean reversion within the weekly trend, avoiding counter-trend trades.
# Works in bull markets by buying dips in uptrend and in bear by selling rallies in downtrend.
# Volume filter ensures only high-conviction signals.

name = "4h_RCI_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rci(close, period):
    """Rank Correlation Index: Spearman correlation between price and time."""
    n = len(close)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        # Rank prices (average ranks for ties)
        sorted_idx = np.argsort(window)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(len(window))
        # Adjust for ties: average rank
        _, inv, counts = np.unique(window, return_inverse=True, return_counts=True)
        if np.any(counts > 1):
            # Compute average rank for each unique value
            sum_ranks = np.bincount(inv, weights=ranks)
            count_each = np.bincount(inv)
            avg_rank = sum_ranks / count_each
            ranks = avg_rank[inv]
        # Rank time: 1,2,...,period
        time_ranks = np.arange(period)
        # Spearman correlation: 1 - (6 * sum(d^2)) / (n*(n^2-1))
        d = ranks - time_ranks
        sum_d2 = np.sum(d * d)
        if period * (period**2 - 1) != 0:
            result[i] = 1 - (6 * sum_d2) / (period * (period**2 - 1))
        else:
            result[i] = 0.0
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for RCI trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly RCI(10) trend ---
    weekly_close = df_weekly['close'].values
    weekly_rci = rci(weekly_close, 10)
    weekly_rci_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rci)
    
    # --- 4h RCI(14) for entry ---
    rci_4h = rci(close, 14)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly RCI(10), 4h RCI(14), and volume MA(20)
    start_idx = max(30, 14, 20)  # weekly needs ~30 bars for stability
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_rci_aligned[i]) or
            np.isnan(rci_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if weekly_rci_aligned[i] > 0 and rci_4h[i] < -80 and vol_spike:
                # Long: weekly uptrend + 4h oversold + volume spike
                signals[i] = 0.25
                position = 1
            elif weekly_rci_aligned[i] < 0 and rci_4h[i] > 80 and vol_spike:
                # Short: weekly downtrend + 4h overbought + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: 4h RCI rises above -20 (recovering from oversold)
                if rci_4h[i] > -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 4h RCI falls below 20 (declining from overbought)
                if rci_4h[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals