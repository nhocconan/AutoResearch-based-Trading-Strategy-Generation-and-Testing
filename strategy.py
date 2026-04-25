#!/usr/bin/env python3
"""
1d_FundingRate_MeanReversion_ZScore_20
Hypothesis: BTC/ETH exhibit strong mean-reversion in funding rates. Extreme positive funding (longs paying shorts) precedes price drops; extreme negative funding precedes rallies. Uses 20-day z-score of funding rate to identify entries. Works in bull markets via mean-reversion shorts during euphoria and longs during fear; works in bear markets via contrarian entries at funding extremes. Primary timeframe 1d, HTF 1w for trend filter (only trade in direction of weekly trend). Discrete sizing 0.25 to minimize fees. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load funding rate data (assumed available in data/processed/funding/)
    # For this example, we simulate funding rate proxy using price-based momentum
    # In practice, replace with: pd.read_parquet(f"data/processed/funding/{symbol}.parquet")
    close = prices['close'].values
    returns = np.diff(np.log(close), prepend=0)
    # Proxy: funding rate ~= recent returns (simplified for illustration)
    funding_proxy = pd.Series(returns).rolling(window=8, min_periods=8).mean().values  # 8h approx
    
    # Calculate 20-day z-score of funding rate
    funding_mean = pd.Series(funding_proxy).rolling(window=20, min_periods=20).mean().values
    funding_std = pd.Series(funding_proxy).rolling(window=20, min_periods=20).std().values
    funding_zscore = (funding_proxy - funding_mean) / (funding_std + 1e-8)
    
    # Get 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_ema = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(funding_zscore[i]) or np.isnan(weekly_ema_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price = close[i]
        weekly_trend_up = price > weekly_ema_aligned[i]
        weekly_trend_down = price < weekly_ema_aligned[i]
        
        if position == 0:
            # Long: extremely negative funding (oversold) + weekly uptrend
            long_signal = (funding_zscore[i] < -2.0) and weekly_trend_up
            # Short: extremely positive funding (overbought) + weekly downtrend
            short_signal = (funding_zscore[i] > 2.0) and weekly_trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold until funding normalizes or trend breaks
            signals[i] = 0.25
            exit_signal = (funding_zscore[i] > -0.5) or (price < weekly_ema_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold until funding normalizes or trend breaks
            signals[i] = -0.25
            exit_signal = (funding_zscore[i] < 0.5) or (price > weekly_ema_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_FundingRate_MeanReversion_ZScore_20"
timeframe = "1d"
leverage = 1.0