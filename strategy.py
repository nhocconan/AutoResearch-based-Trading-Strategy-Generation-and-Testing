#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_ZScore_30d
Hypothesis: Funding rate mean reversion provides edge in BTC/ETH perpetual futures. 
When 30-day funding rate Z-score < -2.0 (extremely negative), go long expecting funding to revert toward mean.
When Z-score > +2.0 (extremely positive), go short expecting funding to revert.
Works in both bull and bear markets as funding extremes often precede price reversals.
Uses 1d timeframe with weekly trend filter (close > 1w EMA50) to avoid counter-trend trades.
Target: 20-40 trades/year with discrete position sizing 0.25 to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load funding rate data (assuming it's available in the same directory structure)
    # For now, we'll simulate funding rate calculation based on price action as proxy
    # In reality, this would load from data/processed/funding/*.parquet
    # Using price-based funding proxy: annualized 8h return deviation from 30d mean
    returns_8h = np.diff(np.log(close), prepend=np.log(close[0])) * 3  # 8h approximation (3*15m if 1d TF, adjusted)
    funding_proxy = pd.Series(returns_8h).rolling(window=30, min_periods=30).mean().values * 365  # annualized
    
    # Calculate 30-day Z-score of funding proxy
    funding_mean = pd.Series(funding_proxy).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_proxy).rolling(window=30, min_periods=30).std().values
    funding_zscore = (funding_proxy - funding_mean) / np.maximum(funding_std, 1e-8)
    
    # Get 1w EMA50 for trend filter (to avoid counter-trend trades in strong trends)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 30d for funding stats, 50d for 1w EMA
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(funding_zscore[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: extremely negative funding Z-score + 1w uptrend (or ranging)
            long_signal = funding_zscore[i] < -2.0 and trend_1w_uptrend
            
            # Short: extremely positive funding Z-score + 1w downtrend (or ranging)
            short_signal = funding_zscore[i] > 2.0 and trend_1w_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: funding reverts to neutral OR trend turns strongly down
            if funding_zscore[i] > -0.5 or not trend_1w_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: funding reverts to neutral OR trend turns strongly up
            if funding_zscore[i] < 0.5 or not trend_1w_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_FundingRate_MeanReversion_ZScore_30d"
timeframe = "1d"
leverage = 1.0