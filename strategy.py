#!/usr/bin/env python3
"""
12h_VolumeSpike_1wTrend_MeanReversion
Hypothesis: In a strong weekly trend (price above/below weekly SMA50), look for mean reversion to the weekly mean on 12h timeframe when accompanied by volume spikes. Buy when price dips below weekly mean with volume spike in uptrend; sell when price rallies above weekly mean with volume spike in downtrend. This captures mean reversion within strong trends, avoiding counter-trend trades. Weekly trend filter ensures alignment with higher timeframe momentum, while volume spikes confirm institutional interest at mean reversion points. Target: 12-37 trades/year per symbol. Works in bull/bear by following weekly trend direction for mean reversion entries.
"""

name = "12h_VolumeSpike_1wTrend_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly SMA50 and mean for trend and reversion levels
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values  # weekly mean for reversion
    
    # Align weekly indicators to 12h timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Volume spike: >2.0x 30-period average (12h) for significant institutional interest
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after SMA50 warmup
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(sma_20_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price below weekly mean (reversion) + weekly uptrend + volume spike
            if (close[i] < sma_20_1w_aligned[i] and 
                close[i] > sma_50_1w_aligned[i] and  # still above long-term trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above weekly mean (reversion) + weekly downtrend + volume spike
            elif (close[i] > sma_20_1w_aligned[i] and 
                  close[i] < sma_50_1w_aligned[i] and  # still below long-term trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back above weekly mean (reversion complete) or breaks weekly trend
            if close[i] > sma_20_1w_aligned[i] or close[i] < sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back below weekly mean (reversion complete) or breaks weekly trend
            if close[i] < sma_20_1w_aligned[i] or close[i] > sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals