#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Stochastic Oscillator with 1-day trend filter and volume confirmation
# Stochastic identifies overbought/oversold conditions. Trend filter ensures trades align with daily direction.
# Volume confirmation filters low-participation moves. Designed for low frequency in 6h timeframe.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

name = "6h_stochastic_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate Stochastic Oscillator (14,3,3) on 6h data
    low_min = pd.Series(low).rolling(window=14, min_periods=14).min()
    high_max = pd.Series(high).rolling(window=14, min_periods=14).max()
    k_percent = 100 * ((close - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=3, min_periods=3).mean()
    k_percent = k_percent.values
    d_percent = d_percent.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after Stochastic warmup
        # Skip if required data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend: close above/below daily EMA20
        daily_uptrend = close[i] > ema20_1d_aligned[i]
        daily_downtrend = close[i] < ema20_1d_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if daily trend turns down or Stochastic becomes overbought
            if not daily_uptrend or k_percent[i] > 80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if daily trend turns up or Stochastic becomes oversold
            if not daily_downtrend or k_percent[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: daily uptrend + Stochastic oversold (<20) + volume confirmation
            if daily_uptrend and k_percent[i] < 20 and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: daily downtrend + Stochastic overbought (>80) + volume confirmation
            elif daily_downtrend and k_percent[i] > 80 and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals