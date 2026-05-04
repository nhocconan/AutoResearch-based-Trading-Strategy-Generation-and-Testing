#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d/1w regime filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) for trend identification on 12h timeframe
# 1d EMA50 for bull/bear regime filter, 1w EMA200 for long-term trend alignment
# Volume confirmation (1.5x 20-period EMA) to filter false breakouts
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total) with discrete sizing (0.25)
# Works in bull markets by going long when Alligator is bullish aligned + price above jaws
# Works in bear markets by going short when Alligator is bearish aligned + price below jaws
# The multi-timeframe regime filter reduces whipsaw during sideways markets

name = "12h_WilliamsAlligator_1dEMA50_1wEMA200_Regime_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for regime filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for regime filter (bull/bear)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w EMA200 for long-term trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Williams Alligator on 12h timeframe (using close prices)
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    # Using EMA as approximation for SMMA (common practice)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: 1.5x 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # Alligator alignment conditions
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bullish_aligned = lips[i] > teeth[i] > jaw[i]
        bearish_aligned = lips[i] < teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish Alligator + price above jaws + volume confirmation + 
            #       price above 1d EMA50 (bullish regime) + price above 1w EMA200 (long-term uptrend)
            if (bullish_aligned and close[i] > jaw[i] and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i] and close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + price below jaws + volume confirmation +
            #        price below 1d EMA50 (bearish regime) + price below 1w EMA200 (long-term downtrend)
            elif (bearish_aligned and close[i] < jaw[i] and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i] and close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR price falls below jaws OR 
            #          regime filter turns bearish
            if (not bullish_aligned or close[i] < jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR price rises above jaws OR
            #           regime filter turns bullish
            if (not bearish_aligned or close[i] > jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals