#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation
# Williams %R measures overbought/oversold levels: values above -20 = overbought, below -80 = oversold
# In strong trends, Williams %R can stay in overbought/oversold territory for extended periods
# Strategy: Buy when Williams %R crosses above -80 from below (oversold bounce) in a weekly uptrend
#           Sell when Williams %R crosses below -20 from above (overbought reversal) in a weekly downtrend
# Weekly trend filter: price above/below weekly EMA20 to avoid counter-trend trades
# Volume confirmation: volume > 1.3x 20-period average to ensure conviction
# Designed to work in both bull and bear markets by following the weekly trend
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 20-period EMA on weekly close for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R crosses above -80 from below
        # AND price above weekly EMA20 (uptrend filter) 
        # AND volume confirmation
        if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
            close[i] > ema20_1w_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R crosses below -20 from above
        # AND price below weekly EMA20 (downtrend filter)
        # AND volume confirmation
        elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
              close[i] < ema20_1w_aligned[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0