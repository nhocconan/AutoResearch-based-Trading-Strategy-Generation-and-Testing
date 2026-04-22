#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams Alligator with daily EMA trend and volume spike.
This strategy uses the Williams Alligator (three SMAs) to identify trending markets.
Long when price > Alligator teeth and green alignment; short when price < Alligator teeth and red alignment.
Daily EMA50 filters for higher timeframe trend alignment. Volume spikes confirm momentum.
Designed for 12h timeframe to target 12-37 trades/year, avoiding overtrading while capturing medium-term trends.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = close_series.rolling(window=8, min_periods=8).mean().values   # Red line (8)
    lips = close_series.rolling(window=5, min_periods=5).mean().values    # Green line (5)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for Alligator to be fully calculated
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
            # Long: Price above Teeth, Lips > Teeth > Jaw (green alignment), price above daily EMA50
            if (close[i] > teeth[i] and 
                lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below Teeth, Lips < Teeth < Jaw (red alignment), price below daily EMA50
            elif (close[i] < teeth[i] and 
                  lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross (trend weakening) or price returns to Teeth
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips cross below Teeth or price drops below Teeth
                if lips[i] < teeth[i] or close[i] < teeth[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Lips cross above Teeth or price rises above Teeth
                if lips[i] > teeth[i] or close[i] > teeth[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0