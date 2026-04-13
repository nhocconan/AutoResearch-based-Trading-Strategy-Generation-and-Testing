# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_1w_MeanReversion_FundingRate
Hypothesis: Uses weekly funding rate mean reversion (Z-score) to capture overextended funding extremes.
In weekly periods when funding rate Z-score < -2.0 (extremely negative), go long; when > +2.0 (extremely positive), go short.
Uses 1d price action for entry timing: wait for price to close above/below 20-period EMA on daily.
Combines funding extreme (contrarian signal) with trend filter (EMA) to avoid picking bottoms/top of strong moves.
Works in both bull and bear markets as funding extremes occur in all regimes.
Target: 20-50 trades/year on 1d (80-200 total over 4 years).
"""

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
    volume = prices['volume'].values
    
    # Get weekly data for funding rate (proxy: using price change as funding proxy)
    # NOTE: In actual implementation, funding rate data would be loaded separately
    # For this simulation, we use weekly price change as a proxy for funding rate extremes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly returns as proxy for funding rate
    weekly_returns = np.diff(close_1w) / close_1w[:-1]
    weekly_returns = np.concatenate([[0], weekly_returns])  # align length
    
    # Calculate Z-score of weekly returns (20-week window)
    returns_series = pd.Series(weekly_returns)
    mean_20w = returns_series.rolling(window=20, min_periods=20).mean()
    std_20w = returns_series.rolling(window=20, min_periods=20).std()
    z_score = (weekly_returns - mean_20w) / std_20w
    z_score = np.where(std_20w == 0, 0, z_score)  # avoid division by zero
    
    # Funding extreme signals: Z-score < -2.0 (long) or > +2.0 (short)
    funding_long_signal = z_score < -2.0
    funding_short_signal = z_score > 2.0
    
    # Get daily data for EMA trend filter and entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period EMA on daily
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all signals to lower timeframe (using index alignment since we're using 1d as primary)
    # For 1d primary timeframe, we can use the data directly with proper indexing
    funding_long_aligned = align_htf_to_ltf(prices, df_1w, funding_long_signal)
    funding_short_aligned = align_htf_to_ltf(prices, df_1w, funding_short_signal)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(funding_long_aligned[i]) or np.isnan(funding_short_aligned[i]) or np.isnan(ema_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = funding_long_aligned[i] and close[i] > ema_20_aligned[i]
        short_entry = funding_short_aligned[i] and close[i] < ema_20_aligned[i]
        
        # Exit conditions: funding extreme passes or price crosses EMA in opposite direction
        long_exit = (not funding_long_aligned[i]) or (close[i] < ema_20_aligned[i])
        short_exit = (not funding_short_aligned[i]) or (close[i] > ema_20_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_MeanReversion_FundingRate"
timeframe = "1d"
leverage = 1.0