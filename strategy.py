# 2025-06-22 | 1d_FundingRateMeanReversion_ZScore_WeeklyTrend
# Hypothesis: Funding rate mean-reversion with weekly trend filter.
# Extreme funding rates (positive = longs paying shorts, negative = shorts paying longs) often precede reversals.
# Z-score of 30-day funding rate identifies extremes: >2 = short signal, <-2 = long signal.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Low trade frequency expected due to extreme threshold (Z>2 or <-2).
# Works in both bull and bear markets as funding extremes occur in all regimes.

name = "1d_FundingRateMeanReversion_ZScore_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data for trend filter (once before loop)
    from mtf_data import get_htf_data, align_htf_to_ltf
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Simulate funding rate data (in practice, load from data/processed/funding/*.parquet)
    # For this example, we'll use a proxy based on price momentum
    # In reality, replace this with actual funding rate data loading
    returns = np.diff(np.log(close), prepend=0)
    # Proxy: funding rate correlates with recent price momentum and volatility
    momentum = np.convolve(returns, np.ones(7)/7, mode='same')  # 7-day momentum
    volatility = np.abs(returns)
    funding_proxy = momentum * (1 + volatility)  # Simplified proxy
    
    # Calculate 30-day Z-score of funding proxy
    if len(funding_proxy) >= 30:
        funding_ma = np.full_like(funding_proxy, np.nan)
        funding_std = np.full_like(funding_proxy, np.nan)
        
        for i in range(29, len(funding_proxy)):
            funding_ma[i] = np.mean(funding_proxy[i-29:i+1])
            funding_std[i] = np.std(funding_proxy[i-29:i+1])
        
        funding_zscore = np.full_like(funding_proxy, np.nan)
        valid = (~np.isnan(funding_std)) & (funding_std > 0)
        funding_zscore[valid] = (funding_proxy[valid] - funding_ma[valid]) / funding_std[valid]
    else:
        funding_zscore = np.full_like(funding_proxy, np.nan)
    
    # Align funding Z-score to daily timeframe (already daily, but for consistency)
    funding_zscore_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), funding_zscore)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 days for Z-score calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(funding_zscore_aligned[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: funding extremely negative (shorts paying longs) AND weekly uptrend
            if (funding_zscore_aligned[i] < -2.0 and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: funding extremely positive (longs paying shorts) AND weekly downtrend
            elif (funding_zscore_aligned[i] > 2.0 and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: funding returns to normal OR trend reversal
            if (funding_zscore_aligned[i] > -0.5 or  # Funding normalized
                close[i] < ema_20_1w_aligned[i]):    # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: funding returns to normal OR trend reversal
            if (funding_zscore_aligned[i] < 0.5 or   # Funding normalized
                close[i] > ema_20_1w_aligned[i]):    # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals