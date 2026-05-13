#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) and close > 1w EMA50 with volume > 1.5x 20-bar average.
# Short when Alligator jaws > teeth > lips (bearish alignment) and close < 1w EMA50 with volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# Williams Alligator identifies trend phases via smoothed medians; 1w EMA50 filters lower-timeframe noise; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
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
    
    # Calculate Williams Alligator from 1d data (jaws=13, teeth=8, lips=5, all smoothed medians)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    jaws = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # Smoothed median, 8-bar shift
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values      # Smoothed median, 5-bar shift
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values       # Smoothed median, 3-bar shift
    
    # Align Alligator lines to lower timeframe (1d -> 1d, identity but using helper for consistency)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 13), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment (jaws < teeth < lips), close > 1w EMA50, volume spike
            if (jaws_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment (jaws > teeth > lips), close < 1w EMA50, volume spike
            elif (jaws_aligned[i] > teeth_aligned[i] and 
                  teeth_aligned[i] > lips_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish Alligator alignment OR volume drops below average
            if (jaws_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i]) or \
               volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish Alligator alignment OR volume drops below average
            if (jaws_aligned[i] < teeth_aligned[i] and 
                teeth_aligned[i] < lips_aligned[i]) or \
               volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals