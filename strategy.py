#!/usr/bin/env python3
# 6h_weekly_ema_crossover_volume_spike_v1
# Hypothesis: 6h strategy using 1w EMA crossover for trend direction with 6h volume spike confirmation.
# Long: Price > 1w EMA(50) AND volume > 2.0x 20-period average AND close > open (bullish candle)
# Short: Price < 1w EMA(50) AND volume > 2.0x 20-period average AND close < open (bearish candle)
# Exit: Opposite EMA crossover (price crosses back below/above 1w EMA(50))
# Uses 6h primary timeframe with 1w HTF for EMA calculation.
# Target: 50-150 total trades over 4 years (12-37/year) to reduce fee drag.
# Weekly EMA provides stable trend filter; volume spike confirms institutional interest.
# Works in both bull and bear markets by following the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_ema_crossover_volume_spike_v1"
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
    open_prices = prices['open'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA(50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w data
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period for EMA
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price crosses below 1w EMA(50)
            if close[i] <= ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above 1w EMA(50)
            if close[i] >= ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price > 1w EMA(50) with volume confirmation and bullish candle
            if close[i] > ema_50_1w_aligned[i] and volume_confirmed and bullish_candle:
                position = 1
                signals[i] = 0.25
            # Short entry: Price < 1w EMA(50) with volume confirmation and bearish candle
            elif close[i] < ema_50_1w_aligned[i] and volume_confirmed and bearish_candle:
                position = -1
                signals[i] = -0.25
    
    return signals