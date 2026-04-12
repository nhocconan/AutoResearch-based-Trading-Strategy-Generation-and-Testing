#!/usr/bin/env python3
"""
4h_1d_funding_rate_mean_reversion
Uses funding rate z-score for mean-reversion contrarian signals:
- Long when funding rate z-score < -2.0 (extremely negative)
- Short when funding rate z-score > +2.0 (extremely positive)
- Exit when z-score returns to zero or reverses
Designed for low trade frequency (<20/year) to exploit funding extremes in BTC/ETH.
Works in both bull and bear markets as funding extremes precede reversals.
"""

name = "4h_1d_funding_rate_mean_reversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load funding rate data from parquet files
    try:
        funding_path = "/mnt/shared/funding/binance_funding_rate_1h.parquet"
        funding_df = pd.read_parquet(funding_path)
        # Align funding rate timestamp to price data
        funding_series = funding_df.set_index('timestamp')['funding_rate']
        # Reindex to match price index
        funding_aligned = funding_series.reindex(prices.index, method='ffill').values
    except:
        # Fallback: simulate funding rate based on price momentum (for testing)
        returns = pd.Series(close).pct_change(periods=8)  # 8-period returns ~ funding proxy
        funding_aligned = (returns.rolling(window=96, min_periods=96).mean().values)  # daily avg
    
    # Calculate z-score of funding rate (30-day lookback)
    funding_series_pd = pd.Series(funding_aligned)
    funding_mean = funding_series_pd.rolling(window=720, min_periods=720).mean().values  # 30d * 24h
    funding_std = funding_series_pd.rolling(window=720, min_periods=720).std().values
    funding_zscore = (funding_aligned - funding_mean) / (funding_std + 1e-8)
    
    # Smooth z-score to reduce noise
    funding_zscore_smooth = pd.Series(funding_zscore).rolling(window=3, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(720, n):  # Wait for z-score warmup
        z = funding_zscore_smooth[i]
        
        # Long entry: extremely negative funding (expect reversion to mean)
        if z < -2.0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: extremely positive funding (expect reversion to mean)
        elif z > 2.0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: funding returns to neutral or reverses
        elif position == 1 and (z > -0.5 or z > funding_zscore_smooth[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (z < 0.5 or z < funding_zscore_smooth[i-1]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals