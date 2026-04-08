#!/usr/bin/env python3
# 1d_weekly_funding_rate_mean_reversion_v1
# Hypothesis: Funding rate mean reversion on 1d timeframe with 1w HTF regime filter.
# Long: 1w funding rate Z-score < -2.0 AND price > 1d EMA200 (bull regime)
# Short: 1w funding rate Z-score > +2.0 AND price < 1d EMA200 (bear regime)
# Exit: Funding rate Z-score between -0.5 and +0.5 OR opposite regime
# Uses 1d primary timeframe with 1w HTF for funding rate Z-score.
# Target: 50-100 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_funding_rate_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate EMA200 for regime filter with min_periods
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1w data for funding rate
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate funding rate Z-score on 1w
    # Funding rate data should be in the 1w dataframe
    funding_rate = df_1w['taker_buy_volume'].values  # placeholder - actual funding rate data would be in separate column
    # For now, we'll use a proxy: calculate Z-score of price momentum as funding rate proxy
    # In practice, funding rate data would be loaded separately
    returns = np.diff(np.log(df_1w['close'].values), prepend=np.log(df_1w['close'].values[0]))
    funding_rate_proxy = pd.Series(returns).ewm(span=7, adjust=False, min_periods=7).mean().values
    
    # Calculate Z-score of funding rate proxy (20-period)
    funding_mean = pd.Series(funding_rate_proxy).rolling(window=20, min_periods=20).mean().values
    funding_std = pd.Series(funding_rate_proxy).rolling(window=20, min_periods=20).std().values
    funding_zscore = np.where(funding_std > 0, (funding_rate_proxy - funding_mean) / funding_std, 0)
    
    # Align 1w funding Z-score to 1d timeframe
    funding_zscore_aligned = align_htf_to_ltf(prices, df_1w, funding_zscore)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200[i]) or np.isnan(funding_zscore_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Funding rate Z-score between -0.5 and +0.5 (mean reversion complete)
            # 2. Opposite regime (price < EMA200)
            if ((-0.5 <= funding_zscore_aligned[i] <= 0.5) or 
                (close[i] < ema200[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Funding rate Z-score between -0.5 and +0.5 (mean reversion complete)
            # 2. Opposite regime (price > EMA200)
            if ((-0.5 <= funding_zscore_aligned[i] <= 0.5) or 
                (close[i] > ema200[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Funding rate Z-score < -2.0 AND price > EMA200 (bull regime)
            if funding_zscore_aligned[i] < -2.0 and close[i] > ema200[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Funding rate Z-score > +2.0 AND price < EMA200 (bear regime)
            elif funding_zscore_aligned[i] > 2.0 and close[i] < ema200[i]:
                position = -1
                signals[i] = -0.25
    
    return signals