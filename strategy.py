#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation.
# Williams Alligator uses three SMAs (jaw=13, teeth=8, lips=5) to identify trend and non-trend markets.
# Long when lips > teeth > jaw (bullish alignment) with volume > 1.5x 20-bar average and price above 1w EMA50.
# Short when lips < teeth < jaw (bearish alignment) with volume > 1.5x 20-bar average and price below 1w EMA50.
# Exit when Alligator lines cross (trend weakening) or volume drops below 0.8x average.
# Designed to capture strong trends in both bull and bear markets while avoiding choppy regimes.
# Discrete sizing 0.25 targets 12-37 trades/year on 12h timeframe.

name = "12h_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm"
timeframe = "12h"
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
    
    lookback = 20  # for volume average
    
    # Calculate Williams Alligator SMAs on 12h timeframe
    # Jaw: SMA(13), Teeth: SMA(8), Lips: SMA(5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    if len(close_1w) < 50:
        ema_50_1w = np.full(len(close_1w), np.nan)
    else:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) with volume spike and price above 1w EMA50
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment) with volume spike and price below 1w EMA50
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator lines cross (lips <= teeth) OR volume drops below 0.8x average
            if lips[i] <= teeth[i] or volume[i] < 0.8 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator lines cross (lips >= teeth) OR volume drops below 0.8x average
            if lips[i] >= teeth[i] or volume[i] < 0.8 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals