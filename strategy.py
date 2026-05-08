#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price above lips
# AND 1d EMA34 rising AND volume > 2x 20-period average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price below lips
# AND 1d EMA34 falling AND volume > 2x 20-period average.
# Exit when Alligator alignment breaks or price crosses lips.
# Williams Alligator uses smoothed moving averages (SMMA) to identify trends.
# The 1d EMA34 filter ensures we trade with the daily trend.
# Volume spike confirms institutional participation.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "4h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan)
    result = np.full_like(data, np.nan)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator on 4h: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: jaw < teeth < lips
            bullish_align = jaw[i] < teeth[i] < lips[i]
            # Bearish alignment: jaw > teeth > lips
            bearish_align = jaw[i] > teeth[i] > lips[i]
            
            # Long conditions: bullish alignment, price above lips, 1d EMA34 rising, volume filter
            long_cond = bullish_align and (close[i] > lips[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: bearish alignment, price below lips, 1d EMA34 falling, volume filter
            short_cond = bearish_align and (close[i] < lips[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment or price below lips
            bearish_align = jaw[i] > teeth[i] > lips[i]
            if bearish_align or (close[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment or price above lips
            bullish_align = jaw[i] < teeth[i] < lips[i]
            if bullish_align or (close[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals