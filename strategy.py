#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d Bollinger squeeze regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price < lower Bollinger Band (20,2), and Bollinger Width < 20th percentile (squeeze regime).
# Short when Williams %R > -20 (overbought), price > upper Bollinger Band (20,2), and Bollinger Width < 20th percentile.
# Uses discrete sizing 0.25. Bollinger squeeze identifies low-volatility regimes where mean reversion works best.
# Williams %R provides timely exhaustion signals. Volume confirmation ensures participation.
# Designed to work in both bull and bear markets by fading extremes during consolidation phases.

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
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for Bollinger Bands and squeeze detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20,2) on 1d
    bb_period = 20
    bb_std = 2
    if len(close_1d) < bb_period:
        return np.zeros(n)
    
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + bb_std * std_20
    lower_bb = sma_20 - bb_std * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate 20th percentile of BB width for squeeze regime (using 50-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_regime = bb_width < bb_width_percentile  # True when in low volatility squeeze
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_regime.astype(float))
    
    # Calculate average volume for confirmation (20-period on 6h)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_wr, bb_period, 50) + 20  # Ensure all indicators are valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wr_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price below lower BB, in squeeze regime, volume spike
            if (wr_aligned[i] < -80 and 
                close[i] < lower_bb_aligned[i] and 
                squeeze_aligned[i] > 0.5 and  # Regime filter
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price above upper BB, in squeeze regime, volume spike
            elif (wr_aligned[i] > -20 and 
                  close[i] > upper_bb_aligned[i] and 
                  squeeze_aligned[i] > 0.5 and  # Regime filter
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (mean reversion) or price reaches middle BB
            if wr_aligned[i] > -50 or close[i] > sma_20[-1] if len(sma_20) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 (mean reversion) or price reaches middle BB
            if wr_aligned[i] < -50 or close[i] < sma_20[-1] if len(sma_20) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals