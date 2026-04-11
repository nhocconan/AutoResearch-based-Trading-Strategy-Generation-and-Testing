#!/usr/bin/env python3
"""
1d_1w_funding_reversion
Strategy: 1-day mean reversion on funding rate extremes with 1-week trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Funding rate extremes predict mean reversion in BTC/ETH. Long when funding is extremely negative (shorts overcrowded), short when extremely positive (longs overcrowded). Uses 1-week price trend as filter to avoid counter-trend trades in strong trends. Designed for low trade frequency (<20/year) to minimize fee drag while capturing persistent funding rate mean reversion observed in BTC/ETH markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_funding_reversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # Load funding rate data (assumed available as column)
    if 'funding_rate' not in prices.columns:
        return np.zeros(n)
    funding = prices['funding_rate'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week EMA for trend filter (50-period)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Funding rate z-score (30-day window)
    funding_mean = pd.Series(funding).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding).rolling(window=30, min_periods=30).std().values
    funding_z = (funding - funding_mean) / funding_std
    funding_z = np.where(funding_std == 0, 0, funding_z)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # start after funding z-score warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(funding_z[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        ema_1w = ema_50_1w_aligned[i]
        fz = funding_z[i]
        
        # Long when funding extremely negative AND price above weekly EMA (avoid strong downtrend)
        long_signal = (fz < -2.0) and (price_close > ema_1w)
        
        # Short when funding extremely positive AND price below weekly EMA (avoid strong uptrend)
        short_signal = (fz > 2.0) and (price_close < ema_1w)
        
        # Exit when funding returns to neutral zone
        exit_long = position == 1 and fz > -0.5
        exit_short = position == -1 and fz < 0.5
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Note: This strategy assumes funding_rate column is available in prices DataFrame.
# If not available, it will return zero signals. In actual implementation,
# funding rate data should be merged from external sources.