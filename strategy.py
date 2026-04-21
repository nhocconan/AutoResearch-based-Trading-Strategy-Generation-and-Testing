#!/usr/bin/env python3
"""
12h_1d_Engulfing_OBV_Divergence
Hypothesis: On 12h timeframe, bullish/bearish engulfing candles at key daily support/resistance levels (prior day high/low) with OBV divergence signal high-probability reversals. Works in bull markets by buying dips at support, in bear markets by selling rallies at resistance. Low frequency due to strict candle pattern + volume divergence requirement.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for prior day high/low
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Prior day high/level (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_high[0] = np.nan  # First day has no prior
    prev_low[0] = np.nan
    
    # Align prior day levels to 12h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_daily, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_daily, prev_low)
    
    # Main timeframe data (12h)
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate OBV
    obv = np.zeros(n)
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # Calculate EMA of OBV for divergence detection
    obv_ema = np.zeros(n)
    if n >= 10:
        obv_ema[9] = np.mean(obv[:10])
        for i in range(10, n):
            obv_ema[i] = 0.18 * obv[i] + 0.82 * obv_ema[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if NaN in critical values
        if np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish engulfing: current green candle engulfs prior red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i] < close[i-1]) and (close[i] > open_price[i-1])
        # Bearish engulfing: current red candle engulfs prior green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i] > close[i-1]) and (close[i] < open_price[i-1])
        
        # OBV divergence: price makes new high/low but OBV doesn't confirm
        bullish_div = (low[i] < low[i-5]) and (obv[i] > obv[i-5]) if i >= 5 else False
        bearish_div = (high[i] > high[i-5]) and (obv[i] < obv[i-5]) if i >= 5 else False
        
        price = close[i]
        prev_high = prev_high_aligned[i]
        prev_low = prev_low_aligned[i]
        
        if position == 0:
            # Long: bullish engulfing at or near prior day low with bullish OBV divergence
            if bullish_engulf and bullish_div and price <= prev_low * 1.005:  # Within 0.5% of prior low
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing at or near prior day high with bearish OBV divergence
            elif bearish_engulf and bearish_div and price >= prev_high * 0.995:  # Within 0.5% of prior high
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches prior day high or bearish engulfing forms
            if price >= prev_high or bearish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches prior day low or bullish engulfing forms
            if price <= prev_low or bullish_engulf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Engulfing_OBV_Divergence"
timeframe = "12h"
leverage = 1.0