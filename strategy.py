#!/usr/bin/env python3
"""
Hypothesis: 4h Funding Rate Mean Reversion + Volume Spike + Chop Filter.
- Uses 4h timeframe (primary) and 8h funding rate data (loaded via get_htf_data)
- Funding rate Z-score (30-period) < -2.0 → long bias, > +2.0 → short bias
- Volume confirmation: current 4h volume > 2.0 * 20-period volume MA
- Chop filter: only trade when Chop(14) > 61.8 (range regime) to avoid trending whipsaws
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: funding mean reversion profits from extremes, chop filter avoids trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 8h funding rate data (Binance funding is every 8h)
    df_8h = get_htf_data(prices, '8h')
    if len(df_8h) < 50:
        return np.zeros(n)
    
    # Funding rate column - assuming it's available in the data
    # If not present, we'll need to simulate or skip - but per instructions, funding data exists
    funding_rate = df_8h['funding_rate'].values if 'funding_rate' in df_8h.columns else np.zeros(len(df_8h))
    
    # Calculate 30-period Z-score of funding rate
    funding_ma = pd.Series(funding_rate).rolling(window=30, min_periods=30).mean().values
    funding_std = pd.Series(funding_rate).rolling(window=30, min_periods=30).std().values
    funding_zscore = np.where(funding_std != 0, (funding_rate - funding_ma) / funding_std, 0)
    
    # Align funding Z-score to 4h timeframe
    funding_zscore_aligned = align_htf_to_ltf(prices, df_8h, funding_zscore)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Chop filter: Chop(14) > 61.8 = range regime (mean revert)
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denominator = atr_14 * 14
    chop = np.where(chop_denominator != 0, 
                    100 * np.log10(highest_high_14 - lowest_low_14) / np.log10(chop_denominator), 
                    50)
    chop_filter = chop > 61.8  # Range regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30, 20, 14)  # Need all indicators warmed up
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(funding_zscore_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: funding extremely negative (mean reversion long) AND volume spike AND range regime
            if funding_zscore_aligned[i] < -2.0 and volume_spike[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: funding extremely positive (mean reversion short) AND volume spike AND range regime
            elif funding_zscore_aligned[i] > +2.0 and volume_spike[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: funding reverts to neutral OR reverse signal
            if funding_zscore_aligned[i] > -0.5 or funding_zscore_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: funding reverts to neutral OR reverse signal
            if funding_zscore_aligned[i] < +0.5 or funding_zscore_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_FundingMeanReversion_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0