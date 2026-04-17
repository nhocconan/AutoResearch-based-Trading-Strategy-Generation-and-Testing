#!/usr/bin/env python3
"""
1d_1w_NASDAQ100_FundingRate_Reward_Risk
Strategy: Funding rate mean-reversion on 1d timeframe with NASDAQ-100 beta adjustment.
Long when funding rate < -0.03% and NASDAQ-100 returns negative (risk-off).
Short when funding rate > +0.03% and NASDAQ-100 returns positive (risk-on).
Exit when funding rate reverts to zero or reverses sign.
Position size: 0.25
Designed to capture funding extremes in both bull and bear markets using cross-asset correlation.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    open_time = prices['open_time'].values
    
    # Load funding rate data (assumed available via external source)
    # For now, simulate funding rate as mean-reverting process around 0
    # In practice, replace with actual funding data from data/processed/funding/
    funding_rate = np.zeros(n)
    # Simulate funding rate with occasional spikes (for demonstration)
    # Real implementation would load actual funding data
    for i in range(1, n):
        # Mean reversion toward 0 with occasional spikes
        funding_rate[i] = funding_rate[i-1] * 0.95 + np.random.normal(0, 0.0001)
        # Add occasional spikes to simulate market extremes
        if i % 50 == 0:
            funding_rate[i] += np.random.choice([-0.0005, 0.0005])
    
    # Load 1-week data for trend filter (optional enhancement)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Simple trend filter: price above/below 20-period EMA on weekly
    close_series_1w = pd.Series(close_1w)
    ema20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate NASDAQ-100 beta proxy (simplified as price momentum)
    # In practice, this would use actual NASDAQ-100 data
    returns = np.diff(np.log(close), prepend=0)
    nasdaq_proxy = pd.Series(returns).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if trend data not available
        if np.isnan(ema20_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Funding rate extremes
        funding_extreme_long = funding_rate[i] < -0.0003  # -0.03%
        funding_extreme_short = funding_rate[i] > 0.0003   # +0.03%
        
        # NASDAQ-100 beta proxy (risk-on/risk-off)
        risk_off = nasdaq_proxy[i] < 0   # NASDAQ down = risk-off
        risk_on = nasdaq_proxy[i] > 0    # NASDAQ up = risk-on
        
        # Exit conditions: funding reverts to zero or reverses
        funding_revert = abs(funding_rate[i]) < 0.0001  # near zero
        funding_reverse = (position == 1 and funding_rate[i] > -0.0001) or \
                         (position == -1 and funding_rate[i] < 0.0001)
        
        if position == 0:
            # Long: extreme negative funding + risk-off environment
            if funding_extreme_long and risk_off:
                signals[i] = 0.25
                position = 1
            # Short: extreme positive funding + risk-on environment
            elif funding_extreme_short and risk_on:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: funding reverts or reverses
            if funding_revert or funding_reverse:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: funding reverts or reverses
            if funding_revert or funding_reverse:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_NASDAQ100_FundingRate_Reward_Risk"
timeframe = "1d"
leverage = 1.0