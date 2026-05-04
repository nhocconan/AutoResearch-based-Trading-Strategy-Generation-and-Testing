#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI + 12h Supertrend combination
# Uses Volume-Weighted RSI (VW-RSI) on 6h chart to identify overbought/oversold conditions with volume confirmation,
# combined with 12h Supertrend as trend filter to avoid counter-trend trades.
# VW-RSI weights price changes by volume, making it more responsive to institutional activity.
# Supertrend on 12h provides robust trend detection that works in both bull and bear markets.
# Designed for 12-30 trades/year (~50-120 total over 4 years) to minimize fee drag.
# Only takes long when VW-RSI < 30 and 12h Supertrend is bullish.
# Only takes short when VW-RSI > 70 and 12h Supertrend is bearish.

name = "6h_VolumeWeightedRSI_12hSupertrend_Combo"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for Supertrend
    tr12h = np.maximum(np.maximum(high_12h[1:] - low_12h[1:], 
                                  np.abs(high_12h[1:] - close_12h[:-1])),
                       np.abs(low_12h[1:] - close_12h[:-1]))
    tr12h = np.concatenate([[np.nan], tr12h])
    atr12h = pd.Series(tr12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 12h Supertrend
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + 3.0 * atr12h
    lower_band_12h = hl2_12h - 3.0 * atr12h
    
    supertrend_12h = np.zeros_like(close_12h)
    supertrend_12h[:] = np.nan
    direction_12h = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend_12h[i-1]):
            supertrend_12h[i] = lower_band_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i] > supertrend_12h[i-1]:
                supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
                direction_12h[i] = 1
            else:
                supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
                direction_12h[i] = -1
    
    # Align Supertrend direction to 6h timeframe (wait for completed 12h bar)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate Volume-Weighted RSI on 6h data
    # VW-RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
    # Gains and Losses are volume-weighted
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weight the gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Calculate average volume-weighted gain and loss over 14 periods
    avg_vol_gain = pd.Series(vol_gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vol_loss != 0, avg_vol_gain / avg_vol_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(direction_12h_aligned[i]) or np.isnan(vw_rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: VW-RSI oversold (<30) AND 12h Supertrend bullish
            if (vw_rsi[i] < 30 and direction_12h_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short conditions: VW-RSI overbought (>70) AND 12h Supertrend bearish
            elif (vw_rsi[i] > 70 and direction_12h_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VW-RSI overbought (>70) OR 12h Supertrend turns bearish
            if (vw_rsi[i] > 70 or direction_12h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VW-RSI oversold (<30) OR 12h Supertrend turns bullish
            if (vw_rsi[i] < 30 or direction_12h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals