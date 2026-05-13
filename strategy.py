#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.5x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Alligator identifies trend emergence; 1d EMA50 filters for higher-timeframe trend; volume confirms conviction.
# Designed to work in both bull (catch trends early) and bear (avoid whipsaws via alignment + trend filter).

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm"
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
    
    lookback = 13  # Williams Alligator default periods
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA (smoothed moving average)
    # SMMA is EMA with alpha = 1/period
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    if len(close) < jaw_period:
        return np.zeros(n)
    
    # Calculate SMMA (Smoothed Moving Average) - equivalent to EMA with alpha=1/period
    jaw = pd.Series(close).ewm(alpha=1/jaw_period, adjust=False, min_periods=jaw_period).mean().values
    teeth = pd.Series(close).ewm(alpha=1/teeth_period, adjust=False, min_periods=teeth_period).mean().values
    lips = pd.Series(close).ewm(alpha=1/lips_period, adjust=False, min_periods=lips_period).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    if len(close_1d) < 50:
        ema_50_1d = np.full(len(close_1d), np.nan)
    else:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaw) AND price > 1d EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaw) AND price < 1d EMA50 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish Alligator alignment OR price crosses below 1d EMA50 OR volume dries up
            if (lips[i] < teeth[i] or teeth[i] < jaw[i] or  # Lost bullish alignment
                close[i] < ema_50_1d_aligned[i] or          # Price below trend
                volume[i] < 0.8 * avg_volume[i]):          # Volume drying up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish Alligator alignment OR price crosses above 1d EMA50 OR volume dries up
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or  # Lost bearish alignment
                close[i] > ema_50_1d_aligned[i] or          # Price above trend
                volume[i] < 0.8 * avg_volume[i]):          # Volume drying up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals