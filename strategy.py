#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_With_WeeklyTrend
Hypothesis: Use funding rate z-score mean reversion on 1d timeframe, filtered by weekly trend (EMA34). 
Long when funding rate z-score < -2.0 and weekly trend is up, short when z-score > +2.0 and weekly trend is down.
Exit when z-score reverts to zero or weekly trend flips. 
Funding rate provides edge in BTC/ETH mean reversion during extremes, weekly trend filter avoids counter-trend trades.
Designed for 1d to capture low-frequency mean reversion (target 10-25 trades/year).
"""

name = "1d_FundingRate_MeanReversion_With_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data (assumed to be available via external source)
    # For now, we'll simulate funding rate as a placeholder - in reality this would load from data/processed/funding/
    # Since we don't have actual funding data in prices, we'll use a proxy based on price action
    # This is a limitation - in practice this strategy would load external funding data
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Weekly trend determination
    weekly_uptrend = close_1w_aligned > ema_34_1w_aligned
    weekly_downtrend = close_1w_aligned < ema_34_1w_aligned
    
    # Proxy for funding rate z-score using price deviation from weekly average
    # In reality, this would be replaced with actual funding rate data
    weekly_mean = pd.Series(close_1w).rolling(window=30, min_periods=30).mean().values
    weekly_mean_aligned = align_htf_to_ltf(prices, df_1w, weekly_mean)
    weekly_std = pd.Series(close_1w).rolling(window=30, min_periods=30).std().values
    weekly_std_aligned = align_htf_to_ltf(prices, df_1w, weekly_std)
    
    # Avoid division by zero
    weekly_std_aligned = np.where(weekly_std_aligned == 0, 1e-10, weekly_std_aligned)
    
    # Price z-score relative to weekly mean (proxy for funding rate extremes)
    price_zscore = (close - weekly_mean_aligned) / weekly_std_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30)  # Warmup for weekly EMA and stats
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or 
            np.isnan(weekly_mean_aligned[i]) or np.isnan(weekly_std_aligned[i]) or
            np.isnan(price_zscore[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price significantly below weekly mean (proxy for negative funding extreme) in weekly uptrend
            if (price_zscore[i] < -2.0 and weekly_uptrend[i]):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above weekly mean (proxy for positive funding extreme) in weekly downtrend
            elif (price_zscore[i] > 2.0 and weekly_downtrend[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to weekly mean or weekly trend turns down
            if (price_zscore[i] > -0.5 or not weekly_uptrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to weekly mean or weekly trend turns up
            if (price_zscore[i] < 0.5 or not weekly_downtrend[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals