#!/usr/bin/env python3
"""
12h_FundingRate_ZScore_Contrarian_HTF
Hypothesis: Uses weekly funding rate z-score (mean-reversion) for contrarian entries on 12h timeframe.
Enter long when weekly funding z-score < -2.0 (extreme negative = oversold short crowding).
Enter short when weekly funding z-score > +2.0 (extreme positive = oversold long crowding).
Add 1d EMA50 trend filter: only long when price > EMA50, only short when price < EMA50.
Exit when funding z-score returns to zero (mean-reversion complete) OR trend reverses.
Funding rate extremes often precede reversals in BTC/ETH; EMA filter avoids fighting strong trends.
Designed for low trade frequency (~20-40/year) to minimize fee drag in bear/ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1w data for funding rate (weekly)
    df_1w = get_htf_data(prices, '1w')
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Funding rate proxy: calculate from price change (actual funding data not available in prices)
    # Use weekly log returns as proxy for funding rate sentiment
    weekly_logret = np.log(df_1w['close'].values / np.roll(df_1w['close'].values, 1))
    weekly_logret[0] = np.nan  # first value invalid
    
    # Calculate z-score of weekly returns (20-week lookback)
    weekly_ret_series = pd.Series(weekly_logret)
    weekly_mean = weekly_ret_series.rolling(window=20, min_periods=20).mean().values
    weekly_std = weekly_ret_series.rolling(window=20, min_periods=20).std().values
    weekly_zscore = (weekly_logret - weekly_mean) / weekly_std
    # Replace infinite/NaN from zero std with 0
    weekly_zscore = np.where((weekly_std == 0) | np.isnan(weekly_zscore), 0, weekly_zscore)
    
    # Align weekly z-score to 12h timeframe
    weekly_zscore_aligned = align_htf_to_ltf(prices, df_1w, weekly_zscore)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w z-score (20), 1d EMA50 (50)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_zscore_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        zscore_val = weekly_zscore_aligned[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry: extreme funding z-score + trend filter
            # Long: z-score < -2.0 (extreme negative) AND price > EMA50 (uptrend bias)
            long_condition = (zscore_val < -2.0) and (close_val > ema_val)
            # Short: z-score > +2.0 (extreme positive) AND price < EMA50 (downtrend bias)
            short_condition = (zscore_val > 2.0) and (close_val < ema_val)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when z-score returns to zero OR trend breaks
            exit_condition = (abs(zscore_val) < 0.5) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when z-score returns to zero OR trend breaks
            exit_condition = (abs(zscore_val) < 0.5) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_FundingRate_ZScore_Contrarian_HTF"
timeframe = "12h"
leverage = 1.0