#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_ZScore
Hypothesis: Funding rate mean reversion works on BTC/ETH in both bull and bear markets.
Extreme positive funding (longs pay shorts) signals overcrowded longs → mean reversion short.
Extreme negative funding (shorts pay longs) signals overcrowded shorts → mean reversion long.
Uses 30-day z-score of funding rate with threshold ±2.0. Weekly trend filter (price vs 1w EMA50)
avoids counter-trend trades. Designed for low trade frequency (10-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load funding rate data (assumed available in data/processed/funding/)
    # Format: funding_rate.csv with columns: funding_time, funding_rate
    try:
        funding_path = "data/processed/funding/funding_rate.csv"
        funding_df = pd.read_csv(funding_path)
        funding_df['funding_time'] = pd.to_datetime(funding_df['funding_time'])
        funding_df.set_index('funding_time', inplace=True)
        
        # Align funding rate to price timestamps
        funding_aligned = pd.Series(index=prices.index, dtype=float)
        for ts in prices.index:
            # Find most recent funding rate (funding updates every 8h)
            mask = funding_df.index <= ts
            if mask.any():
                funding_aligned.loc[ts] = funding_df.loc[mask, 'funding_rate'].iloc[-1]
            else:
                funding_aligned.loc[ts] = 0.0  # neutral if no data
        
        funding = funding_aligned.values
    except:
        # Fallback: if funding data unavailable, use neutral (no signal)
        funding = np.zeros(n)
    
    # Calculate 30-day z-score of funding rate
    funding_series = pd.Series(funding)
    funding_mean = funding_series.rolling(window=30, min_periods=30).mean().values
    funding_std = funding_series.rolling(window=30, min_periods=30).std().values
    funding_zscore = np.where(funding_std > 0, (funding - funding_mean) / funding_std, 0.0)
    
    # Weekly trend filter: price vs 1w EMA50
    try:
        from mtf_data import get_htf_data, align_htf_to_ltf
        df_1w = get_htf_data(prices, '1w')
        close_1w = df_1w['close'].values
        ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    except:
        # Fallback: use 50-period EMA on 1d data if 1w unavailable
        ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema50_1w_aligned = ema50  # approximate
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need funding z-score (30), EMA50 (50)
    start_idx = max(30, 50)
    
    for i in range(start_idx, n):
        zscore = funding_zscore[i]
        ema50 = ema50_1w_aligned[i]
        close_val = close[i]
        
        # Skip if data not ready
        if np.isnan(zscore) or np.isnan(ema50):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long when funding extremely negative (shorts crowded) AND price above weekly EMA
            if zscore < -2.0 and close_val > ema50:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short when funding extremely positive (longs crowded) AND price below weekly EMA
            elif zscore > 2.0 and close_val < ema50:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: funding returns to neutral OR price breaks below weekly EMA
            if zscore > -0.5 or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: funding returns to neutral OR price breaks above weekly EMA
            if zscore < 0.5 or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_FundingRate_MeanReversion_ZScore"
timeframe = "1d"
leverage = 1.0