#!/usr/bin/env python3
# 1d_Wilcoxon_Rank_Sum_Trend_Filter
# Hypothesis: Daily Wilcoxon rank-sum test on price changes vs median to detect persistent trends.
# Non-parametric test avoids distribution assumptions, works in bull/bear via trend filter.
# Uses 1-week trend filter (EMA50) and volume confirmation to avoid whipsaws.
# Targets 15-25 trades/year on 1d timeframe with decisive trend signals.

name = "1d_Wilcoxon_Rank_Sum_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def wilcoxon_rank_sum(x, y):
    """Manual Wilcoxon rank-sum (Mann-Whitney U) for two samples."""
    combined = np.concatenate([x, y])
    n_x = len(x)
    n_y = len(y)
    
    # Rank the combined data
    ranked = stats.rankdata(combined)
    rank_x = ranked[:n_x]
    
    # Calculate U statistic
    u_x = np.sum(rank_x) - n_x * (n_x + 1) / 2
    u_y = n_x * n_y - u_x
    
    # Use smaller U
    u = min(u_x, u_y)
    
    # For large samples, approximate z-score
    if n_x > 20 and n_y > 20:
        mu = n_x * n_y / 2
        sigma = np.sqrt(n_x * n_y * (n_x + n_y + 1) / 12)
        z = (u - mu) / sigma
        return z
    else:
        return u  # Return U for small samples

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily price changes (returns)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    
    # Wilcoxon rank-sum test: recent returns vs historical median
    lookback = 20  # Days for historical comparison
    test_window = 10  # Days to test for trend
    
    # Precompute Wilcoxon statistics
    wilcoxon_stats = np.full(n, np.nan)
    
    for i in range(lookback + test_window, n):
        # Historical returns (reference distribution)
        hist_start = i - lookback - test_window
        hist_end = i - test_window
        historical_returns = returns[hist_start:hist_end]
        
        # Recent returns to test
        recent_returns = returns[i - test_window:i]
        
        if len(historical_returns) >= 10 and len(recent_returns) >= 5:
            try:
                # Use scipy's mannwhitneyu for efficiency
                stat, pval = stats.mannwhitneyu(
                    recent_returns, historical_returns, 
                    alternative='two-sided'
                )
                # Convert to z-score approximation for signal strength
                n1, n2 = len(recent_returns), len(historical_returns)
                mu = n1 * n2 / 2
                sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
                z = (stat - mu) / sigma
                wilcoxon_stats[i] = z
            except:
                wilcoxon_stats[i] = 0.0
    
    # 1-week EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume / 20-day average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback + test_window, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wilcoxon_stats[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: positive Wilcoxon z-score (recent returns > historical) AND uptrend AND volume spike
            if (wilcoxon_stats[i] > 1.0 and  # Significant positive trend
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: negative Wilcoxon z-score (recent returns < historical) AND downtrend AND volume spike
            elif (wilcoxon_stats[i] < -1.0 and  # Significant negative trend
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakness (Wilcoxon near zero) OR trend reversal
            if wilcoxon_stats[i] < 0.5 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakness (Wilcoxon near zero) OR trend reversal
            if wilcoxon_stats[i] > -0.5 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals