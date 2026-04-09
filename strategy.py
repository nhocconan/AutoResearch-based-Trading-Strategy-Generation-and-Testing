#!/usr/bin/env python3
# 1d_funding_rate_mean_reversion_v1
# Hypothesis: Funding rate mean reversion on 1d timeframe with 1w HTF trend filter.
# Extreme funding rates (Z-score > 2.0 or < -2.0 over 30d window) signal overextended
# perpetual futures sentiment. Counter-trend positions taken when funding extreme
# coincides with 1w trend alignment (price above/below 1w EMA50). Works in both bull
# and bear markets by fading funding extremes while respecting higher timeframe trend.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_funding_rate_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Funding rate data (assumed available in prices DataFrame)
    # If not available, fallback to price-based proxy
    if 'funding_rate' in prices.columns:
        funding = prices['funding_rate'].values
    else:
        # Proxy: calculate funding rate basis from price deviation from VWAP
        # This is a simplified proxy - in reality funding rate comes from separate data
        vwap = (prices['close'] * prices['volume']).cumsum() / prices['volume'].cumsum()
        funding = (close - vwap.values) / vwap.values * 0.01  # scaled proxy
    
    # Funding rate Z-score (30-day window)
    funding_series = pd.Series(funding)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean().values
    funding_std = funding_series.rolling(window=30, min_periods=30).std().values
    funding_z = (funding - funding_mean) / (funding_std + 1e-8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(funding_z[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Extreme funding rate thresholds
        funding_extreme_long = funding_z[i] < -2.0  # Very negative funding = long opportunity
        funding_extreme_short = funding_z[i] > 2.0   # Very positive funding = short opportunity
        
        # 1w trend filter: price above/below 1w EMA50
        price_above_1w_trend = close[i] > ema_50_1w_aligned[i]
        price_below_1w_trend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: funding normalizes OR price crosses below 1w EMA
            if funding_z[i] > -0.5 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: funding normalizes OR price crosses above 1w EMA
            if funding_z[i] < 0.5 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if funding_extreme_long and price_above_1w_trend:
                position = 1
                signals[i] = 0.25
            elif funding_extreme_short and price_below_1w_trend:
                position = -1
                signals[i] = -0.25
    
    return signals