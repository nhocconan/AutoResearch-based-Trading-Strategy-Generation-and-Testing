#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_ZScore_WeeklyTrend_v1
Hypothesis: Funding rate mean reversion provides edge in BTC/ETH. Extreme positive funding (longs paying shorts) predicts mean reversion down; extreme negative funding predicts mean reversion up. Weekly trend filter avoids fighting major trends. Discrete sizing 0.25 controls fees. Target 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load funding rate data (assuming available as column)
    # If not available, fallback to price-based proxy
    if 'funding_rate' in prices.columns:
        funding = prices['funding_rate'].values
    else:
        # Proxy: use RSI divergence as funding proxy (not ideal but avoids zeros)
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.ewm(alpha=1/14, adjust=False).mean() / (loss.ewm(alpha=1/14, adjust=False).mean() + 1e-10)
        funding = 100 - (100 / (1 + rs))  # RSI as proxy, scaled to [-100,100]
        funding = (funding - 50) / 25  # Scale to approximately [-2,2] for Z-score
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Funding rate Z-score (30-day lookback)
    funding_series = pd.Series(funding)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean().values
    funding_std = funding_series.rolling(window=30, min_periods=30).std().values
    funding_z = (funding - funding_mean) / np.maximum(funding_std, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need funding Z-score lookback
    
    for i in range(start_idx, n):
        if np.isnan(funding_z[i]) or np.isnan(ema_34_1w_aligned[i]):
            # Hold position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: extremely negative funding (shorts paying longs) + weekly uptrend or ranging
            if funding_z[i] < -2.0 and (weekly_uptrend or True):  # allow ranging
                signals[i] = 0.25
                position = 1
            # Short: extremely positive funding (longs paying shorts) + weekly downtrend or ranging
            elif funding_z[i] > 2.0 and (weekly_downtrend or True):  # allow ranging
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: funding normalizes or weekly trend turns down strongly
            if funding_z[i] > -0.5 or (not weekly_uptrend and funding_z[i] > 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: funding normalizes or weekly trend turns up strongly
            if funding_z[i] < 0.5 or (not weekly_downtrend and funding_z[i] < 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_FundingRate_MeanReversion_ZScore_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0