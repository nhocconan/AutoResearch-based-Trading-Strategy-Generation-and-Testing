#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Mean Reversion with 12h EMA200 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 12h EMA200 (bullish trend), and volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought), price < 12h EMA200 (bearish trend), and volume > 1.5x average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# EMA200 ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws in bear markets.
# Williams %R provides timely mean reversal signals in ranging conditions.
# Volume confirmation validates the strength of the mean reversion move.

name = "6h_WilliamsR_MeanReversion_12hEMA200_VolumeConfirm"
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
    
    lookback = 14  # for Williams %R
    vol_lookback = 20  # for volume average
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(200) on 12h close
    if len(close_12h) < 200:
        ema_200_12h = np.full(len(close_12h), np.nan)
    else:
        ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA to 6h timeframe (wait for 12h bar to close)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate Williams %R on 6h data
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=vol_lookback, min_periods=vol_lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_200_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), bullish 12h EMA trend, volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_200_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), bearish 12h EMA trend, volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals