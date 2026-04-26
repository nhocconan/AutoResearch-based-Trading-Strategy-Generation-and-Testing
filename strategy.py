#!/usr/bin/env python3
"""
1d_FundingRateMeanReversion_Zscore_30d
Hypothesis: Funding rate mean reversion on 1d timeframe using 30-day z-score.
When funding rate is extremely negative (< -2.0 z-score), go long expecting positive funding reversion.
When funding rate is extremely positive (> +2.0 z-score), go short expecting negative funding reversion.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
Incorporates 1w trend filter to avoid fighting the major trend. Only takes longs in 1w uptrend, shorts in 1w downtrend.
Adds volume confirmation (>1.5x average volume) to ensure participation.
Works in both bull and bear markets because funding extremes occur in all regimes and mean reversion is statistically robust.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load funding rate data (assuming it's available in prices DataFrame)
    # If not available, we'll skip this strategy (but it should be available per instructions)
    if 'funding_rate' not in prices.columns:
        return np.zeros(n)
    
    funding = prices['funding_rate'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 30-day z-score of funding rate
    funding_series = pd.Series(funding)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean().values
    funding_std = funding_series.rolling(window=30, min_periods=30).std().values
    
    # Avoid division by zero
    funding_std = np.where(funding_std == 0, 1e-10, funding_std)
    funding_zscore = (funding - funding_mean) / funding_std
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for z-score, 20 for EMA and volume)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        zscore = funding_zscore[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_20_1w_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(zscore) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: extremely negative funding + 1w uptrend + volume confirmation
        long_condition = (zscore < -2.0) and (close[i] > ema_val) and volume_confirmed
        # Short logic: extremely positive funding + 1w downtrend + volume confirmation
        short_condition = (zscore > 2.0) and (close[i] < ema_val) and volume_confirmed
        
        # Exit logic: funding returns to normal or trend reversal
        exit_long = (zscore > -0.5) or (close[i] < ema_val)
        exit_short = (zscore < 0.5) or (close[i] > ema_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_FundingRateMeanReversion_Zscore_30d"
timeframe = "1d"
leverage = 1.0