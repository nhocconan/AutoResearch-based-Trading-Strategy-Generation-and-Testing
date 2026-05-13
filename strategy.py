#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d Bollinger Band squeeze regime filter.
# Long when Williams %R(14) < -80 (oversold) and price < BB lower band (mean reversion setup)
# in a low volatility regime (BB width < 30th percentile of 50-period lookback).
# Short when Williams %R(14) > -20 (overbought) and price > BB upper band
# in a low volatility regime.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses Bollinger Band width percentile as regime filter to avoid whipsaws in high volatility.
# Discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# Williams %R is effective at identifying exhaustion points, and BB squeeze filters for
# mean-reversion favorable conditions. Works in both bull and bear markets by fading extremes.

name = "6h_WilliamsR_MeanReversion_1dBB_Squeeze_Regime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Williams %R and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback_wr = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    wr_1d = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate Bollinger Bands(20,2) on 1d close
    bb_period = 20
    bb_std = 2
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + bb_std * std_1d
    lower_bb = sma_1d - bb_std * std_1d
    bb_width = upper_bb - lower_bb
    
    # Calculate BB width percentile regime filter (30th percentile lookback = 50 periods)
    regime_lookback = 50
    bb_width_percentile = pd.Series(bb_width).rolling(window=regime_lookback, min_periods=regime_lookback).quantile(0.30).values
    low_volatility_regime = bb_width < bb_width_percentile
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    low_volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, low_volatility_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(wr_1d_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(low_volatility_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) AND price below lower BB AND low volatility regime
            if (wr_1d_aligned[i] < -80 and 
                close[i] < lower_bb_aligned[i] and 
                low_volatility_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) AND price above upper BB AND low volatility regime
            elif (wr_1d_aligned[i] > -20 and 
                  close[i] > upper_bb_aligned[i] and 
                  low_volatility_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion complete)
            if wr_1d_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion complete)
            if wr_1d_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals