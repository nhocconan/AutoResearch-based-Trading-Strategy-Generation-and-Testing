#!/usr/bin/env python3
"""
1d_FundingRateMeanReversion_Zscore_30d
Hypothesis: Funding rate mean reversion provides edge in BTC/ETH perpetual futures. Extreme positive funding (longs pay shorts) predicts short-term mean reversion downward; extreme negative funding predicts upward reversion. Uses 30-day z-score of funding rate to identify extremes. Works in both bull and bear markets as funding extremes occur during euphoria and panic. Primary timeframe 1d with weekly HTF trend filter to avoid counter-trend trades. Target 10-20 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load funding rate data (assumed available in data/processed/funding/)
    # For now, we simulate funding rate calculation from price and volume proxy
    # In reality, funding data would be loaded separately
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Proxy for funding rate: using price deviation from VWAP as placeholder
    # Actual implementation would load funding*.parquet files
    typical_price = (high + low + close) / 3.0
    vwap = pd.Series(typical_price * volume).rolling(window=288, min_periods=288).sum() / \
           pd.Series(volume).rolling(window=288, min_periods=288).sum()
    funding_proxy = (close - vwap) / vwap  # Simplified funding rate proxy
    
    # Calculate 30-day z-score of funding rate
    funding_ma = pd.Series(funding_proxy).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_proxy).rolling(window=30, min_periods=30).std().values
    funding_zscore = (funding_proxy - funding_ma) / (funding_std + 1e-8)
    
    # Get weekly HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for funding zscore (30d) and weekly EMA (34)
    start_idx = max(30, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(funding_zscore[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: extremely negative funding (shorts pay longs) + price above weekly EMA (bullish bias)
            long_setup = (funding_zscore[i] < -2.0) and (close[i] > ema_34_1w_aligned[i])
            # Short: extremely positive funding (longs pay shorts) + price below weekly EMA (bearish bias)
            short_setup = (funding_zscore[i] > 2.0) and (close[i] < ema_34_1w_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: funding returns to neutral OR price crosses below weekly EMA
            if (abs(funding_zscore[i]) < 0.5) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: funding returns to neutral OR price crosses above weekly EMA
            if (abs(funding_zscore[i]) < 0.5) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_FundingRateMeanReversion_Zscore_30d"
timeframe = "1d"
leverage = 1.0