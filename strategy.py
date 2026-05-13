#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d Bollinger Band squeeze filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price touches lower BB, and volume > 1.5x average during BB squeeze (BB width < 20th percentile).
# Short when Williams %R > -20 (overbought), price touches upper BB, and volume > 1.5x average during BB squeeze.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Williams %R identifies exhaustion, Bollinger Band squeeze indicates low volatility primed for breakout/reversal,
# volume confirmation ensures participation. Works in bull markets via bounces from support and in bear markets via bounces from resistance.

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
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2) on 1d data
    bb_period = 20
    bb_std = 2
    if len(close_1d) < bb_period:
        bb_upper = np.full_like(close_1d, np.nan)
        bb_lower = np.full_like(close_1d, np.nan)
        bb_middle = np.full_like(close_1d, np.nan)
        bb_width = np.full_like(close_1d, np.nan)
    else:
        bb_middle = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
        bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
        bb_upper = bb_middle + bb_std * bb_std_dev
        bb_lower = bb_middle - bb_std * bb_std_dev
        bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile (20th) for squeeze detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback_wr + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_squeeze_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        bb_squeeze_bool = bb_squeeze_aligned[i] > 0.5
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price touches lower BB, BB squeeze, volume spike
            if (williams_r[i] < -80 and 
                close[i] <= bb_lower_aligned[i] and 
                bb_squeeze_bool and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price touches upper BB, BB squeeze, volume spike
            elif (williams_r[i] > -20 and 
                  close[i] >= bb_upper_aligned[i] and 
                  bb_squeeze_bool and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (recovering) OR price touches upper BB
            if (williams_r[i] > -50) or (close[i] >= bb_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (recovering) OR price touches lower BB
            if (williams_r[i] < -50) or (close[i] <= bb_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals