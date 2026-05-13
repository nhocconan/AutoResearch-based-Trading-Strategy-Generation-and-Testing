#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d Bollinger Band squeeze filter and volume confirmation.
# Long when Williams %R < -80 (oversold) + price below lower BB(20,2) + BB width at 20-period low + volume > 1.5x average.
# Short when Williams %R > -20 (overbought) + price above upper BB(20,2) + BB width at 20-period low + volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Williams %R identifies exhaustion, BB squeeze indicates low volatility primed for expansion, volume confirms participation.
# Works in bull markets via mean reversion from oversold and in bear markets via mean reversion from overbought.

name = "6h_WilliamsR_MeanReversion_1dBBSqueeze_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 6h data
    lookback_wr = 14
    if n < lookback_wr:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20,2) on 1d data
    bb_period = 20
    if len(close_1d) < bb_period:
        return np.zeros(n)
    
    ma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Check if BB width is at 20-period low (squeeze condition)
    bb_width_low = pd.Series(bb_width).rolling(window=bb_period, min_periods=bb_period).min().values
    bb_squeeze = (bb_width <= bb_width_low * 1.1)  # within 10% of the low
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    ma_20_aligned = align_htf_to_ltf(prices, df_1d, ma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_wr + bb_period, 20)  # Ensure sufficient data for all indicators
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ma_20_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold + price below lower BB + BB squeeze + volume spike
            if (williams_r[i] < -80 and 
                close[i] < lower_bb_aligned[i] and 
                bb_squeeze_aligned[i] > 0.5 and  # Boolean aligned as 0.0 or 1.0
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought + price above upper BB + BB squeeze + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] > upper_bb_aligned[i] and 
                  bb_squeeze_aligned[i] > 0.5 and
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (mean reversion complete) or price breaks above upper BB
            if williams_r[i] > -50 or close[i] > upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 (mean reversion complete) or price breaks below lower BB
            if williams_r[i] < -50 or close[i] < lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals