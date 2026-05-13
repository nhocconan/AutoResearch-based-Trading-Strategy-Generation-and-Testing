#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with weekly Bollinger Band regime filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > weekly BB middle, and volume > 1.5x average.
# Short when Williams %R > -20 (overbought), price < weekly BB middle, and volume > 1.5x average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Williams %R identifies exhaustion points; weekly BB regime ensures mean reversion in ranging markets and avoids strong trends.
# Volume confirmation reduces false signals. Works in bull markets via mean reversion at extremes and in bear markets via similar mechanics.

name = "6h_WilliamsR_MeanReversion_WeeklyBB_Regime_VolumeConfirm"
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
    
    lookback = 14  # for Williams %R and volume average
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get weekly Bollinger Bands (20, 2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    bb_middle = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align weekly BB to 6h timeframe (wait for weekly bar to close)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(bb_middle_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price above weekly BB middle, volume spike
            if (williams_r[i] < -80 and 
                close[i] > bb_middle_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price below weekly BB middle, volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < bb_middle_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R rises above -50 (mean reversion complete) OR price hits weekly BB upper
            if (williams_r[i] > -50 or 
                close[i] > bb_upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R falls below -50 (mean reversion complete) OR price hits weekly BB lower
            if (williams_r[i] < -50 or 
                close[i] < bb_lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals