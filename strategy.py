#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d Funding Rate Z-Score Mean Reversion.
- Williams %R (14) identifies overbought/oversold conditions on 6h chart.
- Extreme readings (%R < -90 for long, %R > -10 for short) signal exhaustion.
- 1d funding rate z-score (30d window) provides market sentiment extreme filter.
- Long when %R < -90 AND funding z-score < -1.5 (oversold + negative funding sentiment).
- Short when %R > -10 AND funding z-score > +1.5 (overbought + positive funding sentiment).
- Volume confirmation (> 1.5x 20-period average) ensures institutional participation.
- Uses discrete position sizing (0.25) to limit drawdown and minimize fee churn.
- Works in bull/bear markets via funding rate sentiment as contrarian indicator.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Get 1d data ONCE before loop for funding rate z-score
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Funding rate data (assuming available in prices DataFrame as 'funding_rate')
    # If not available, we'll skip this filter and rely on Williams %R + volume
    if 'funding_rate' in prices.columns:
        funding_rate = prices['funding_rate'].values
        funding_ma = pd.Series(funding_rate).rolling(window=30, min_periods=30).mean().values
        funding_std = pd.Series(funding_rate).rolling(window=30, min_periods=30).std().values
        funding_z = (funding_rate - funding_ma) / (funding_std + 1e-10)
        # Align funding z-score to 6h (already aligned as it's from prices)
        funding_z_aligned = funding_z  # already in 6h timeframe
    else:
        # Fallback: use price-based mean reversion if funding not available
        funding_z_aligned = np.zeros(n)  # neutral, won't trigger
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            (np.isnan(funding_z_aligned[i]) and 'funding_rate' in prices.columns)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Williams %R extreme + funding rate sentiment
                wr_long = williams_r[i] < -90
                wr_short = williams_r[i] > -10
                
                if 'funding_rate' in prices.columns:
                    fund_long = funding_z_aligned[i] < -1.5
                    fund_short = funding_z_aligned[i] > 1.5
                    
                    # Long: oversold + negative funding sentiment
                    if wr_long and fund_long:
                        signals[i] = 0.25
                        position = 1
                    # Short: overbought + positive funding sentiment
                    elif wr_short and fund_short:
                        signals[i] = -0.25
                        position = -1
                else:
                    # Fallback: Williams %R extremes only (less reliable but functional)
                    if wr_long:
                        signals[i] = 0.25
                        position = 1
                    elif wr_short:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R returns above -50 (reversal from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns below -50 (reversal from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dFundingZ_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0