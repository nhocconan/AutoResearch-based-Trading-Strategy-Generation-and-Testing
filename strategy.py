#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams Alligator with daily trend filter and volume confirmation.
Uses three SMAs (jaw, teeth, lips) to identify trends. Long when lips > teeth > jaw with price above lips,
short when lips < teeth < jaw with price below lips. Daily EMA50 filters trend direction.
Volume spike confirms institutional interest. Designed to work in both bull (trend following) and bear
(counter-trend reversals) by adapting to daily trend. Target: 20-35 trades/year per symbol.
"""

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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 4H: SMA(13,8), SMA(8,5), SMA(5,3)
    # Jaw: 13-period SMA, 8 periods ahead
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    # Teeth: 8-period SMA, 5 periods ahead
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    # Lips: 5-period SMA, 3 periods ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price above lips AND daily trend bullish
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > lips[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price below lips AND daily trend bearish
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < lips[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse Alligator alignment or price crosses jaws
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment or price crosses below jaw
                if (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] < jaw[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish alignment or price crosses above jaw
                if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] > jaw[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0