#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume spike.
# The Alligator (Jaw, Teeth, Lips) identifies trend absence/presence. 
# Long when Lips > Teeth > Jaw (bullish alignment) with 1w uptrend and volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) with 1w downtrend and volume spike.
# Uses Williams Alligator (SMMA based) to avoid whipsaws in ranging markets.
# Designed for 12h timeframe to target 12-37 trades/year, avoiding fee drag.
# Works in bull markets (follow 1w uptrend) and bear markets (follow 1w downtrend).
name = "12h_Williams_Alligator_1wEMA_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Williams Alligator on 12h data (SMMA = Smoothed Moving Average)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period, shift):
        # Smoothed Moving Average: similar to EMA but with different smoothing
        if len(arr) < period + shift:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period + shift - 1] = np.mean(arr[shift:period + shift])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period + shift, len(arr)):
            result[i] = (result[i-1] * (period - 1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    # 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 50-period EMA (high threshold for fewer trades)
    vol_ema50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (2.0 * vol_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient data for Alligator
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + 1w uptrend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                price > ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + 1w downtrend + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  price < ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bullish alignment breaks or trend reverses
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bearish alignment breaks or trend reverses
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals