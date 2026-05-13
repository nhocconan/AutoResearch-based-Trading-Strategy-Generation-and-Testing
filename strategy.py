#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d Bollinger Band squeeze filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND price < lower Bollinger Band(20,2) AND volume > 1.5x average.
# Short when Williams %R > -20 (overbought) AND price > upper Bollinger Band(20,2) AND volume > 1.5x average.
# Uses Bollinger Band squeeze (bandwidth < 20th percentile) as regime filter to avoid whipsaws in strong trends.
# Williams %R provides timely mean reversion signals in ranging markets, while Bollinger squeeze identifies low volatility periods conducive to mean reversion.
# Volume confirmation ensures participation. Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull markets via buying dips and in bear markets via selling rallies within the larger trend.

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
    
    # Calculate Williams %R(14) on 6h data
    lookback_wr = 14
    if n < lookback_wr:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Bollinger Bands(20,2) on 1d data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    if len(close_1d) < 20:
        return np.zeros(n)
    
    # Basis: 20-period SMA
    basis = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    dev = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    upper_bb = basis + 2 * dev
    lower_bb = basis - 2 * dev
    # Bollinger Band Width
    bb_width = (upper_bb - lower_bb) / basis
    # Handle division by zero
    bb_width = np.where(basis == 0, 0, bb_width)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window to avoid look-ahead)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):
        bb_width_percentile[i] = np.percentile(bb_width[:i+1], 20)
    
    # Bollinger Band Squeeze: width < 20th percentile
    bb_squeeze = bb_width < bb_width_percentile
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))  # bool to float for alignment
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for BB
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(bb_squeeze_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold Williams %R AND price below lower BB AND BB squeeze AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] < lower_bb_aligned[i] and 
                bb_squeeze_aligned[i] > 0.5 and  # Convert back to bool (aligned as float)
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought Williams %R AND price above upper BB AND BB squeeze AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] > upper_bb_aligned[i] and 
                  bb_squeeze_aligned[i] > 0.5 and  # Convert back to bool (aligned as float)
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns above -50 (mean reversion complete) OR price crosses above basis
            basis_aligned = align_htf_to_ltf(prices, df_1d, basis)
            if np.isnan(basis_aligned[i]):
                signals[i] = 0.25
            elif williams_r_aligned[i] > -50 or close[i] > basis_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns below -50 (mean reversion complete) OR price crosses below basis
            basis_aligned = align_htf_to_ltf(prices, df_1d, basis)
            if np.isnan(basis_aligned[i]):
                signals[i] = -0.25
            elif williams_r_aligned[i] < -50 or close[i] < basis_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals