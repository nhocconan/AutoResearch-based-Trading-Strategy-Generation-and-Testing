#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Volume Spike + 1w Trend Filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
# Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish).
# Entry confirmed by 1d volume > 2x 20-period median (institutional interest).
# Exit when Alligator lines re-intertwine (market returns to range).
# Designed to catch trends in both bull and bear markets while avoiding whipsaws in ranges.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week trend filter: close > EMA200 = bullish bias, close < EMA200 = bearish bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Williams Alligator on 4h median price (hlc3)
    hlc3 = (high + low + close) / 3.0
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # Smoothed with 8-bar shift
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # Smoothed with 5-bar shift
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # Smoothed with 3-bar shift
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for Alligator shifts
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Long: Bullish alignment, price above weekly EMA200, volume spike
        if bullish and close[i] > ema_200_1w_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: Bearish alignment, price below weekly EMA200, volume spike
        elif bearish and close[i] < ema_200_1w_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: Alligator lines re-intertwine (not clearly aligned) OR price crosses weekly EMA200
        elif i > 0 and signals[i-1] != 0:
            prev_signal = signals[i-1]
            # Exit long if alignment breaks or price drops below weekly EMA
            if prev_signal == 0.25 and not (lips[i] > teeth[i] > jaw[i]) or close[i] <= ema_200_1w_aligned[i]:
                signals[i] = 0.0
            # Exit short if alignment breaks or price rises above weekly EMA
            elif prev_signal == -0.25 and not (lips[i] < teeth[i] < jaw[i]) or close[i] >= ema_200_1w_aligned[i]:
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsAlligator_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0