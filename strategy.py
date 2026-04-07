#!/usr/bin/env python3
"""
4h_donchian_breakout_volume_v3
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian upper band with above-average volume and 12h trend up (close > SMA50), enter short when price breaks below 20-period Donchian lower band with above-average volume and 12h trend down (close < SMA50). Exit when price crosses the 20-period Donchian mid-band. Uses volume confirmation to avoid false breakouts and trend filter to align with higher timeframe momentum. Designed for 20-40 trades/year to minimize fee decay while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) < 20:
        return np.zeros(n)
    
    # Upper band: highest high of last 20 periods
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h SMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    sma_50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    sma_50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(sma_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle band
            if close[i] < donch_mid[i] and close[i-1] >= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle band
            if close[i] > donch_mid[i] and close[i-1] <= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian upper band with 12h uptrend
                if close[i] > donch_high[i] and close[i-1] <= donch_high[i-1] and close[i] > sma_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band with 12h downtrend
                elif close[i] < donch_low[i] and close[i-1] >= donch_low[i-1] and close[i] < sma_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals