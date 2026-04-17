#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA100 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; reversals from extremes
# work in both bull (pullbacks in uptrend) and bear (bounces in downtrend).
# Uses 1d EMA100 for trend filter and volume spike for confirmation.
# Target: 12-37 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 1d EMA100 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema100_1d = close_1d_series.ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d indicators to 12h timeframe
    williams_r_12h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema100_12h = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need sufficient data for Williams %R and EMA100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_12h[i]) or 
            np.isnan(ema100_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA100
        price_above_ema = close[i] > ema100_12h[i]
        price_below_ema = close[i] < ema100_12h[i]
        
        # Williams %R levels: oversold < -80, overbought > -20
        oversold = williams_r_12h[i] < -80
        overbought = williams_r_12h[i] > -20
        
        if position == 0:
            # Long: Williams %R rises from oversold (<-80) with volume and above EMA100
            if (williams_r_12h[i] > williams_r_12h[i-1] and  # rising from oversold
                williams_r_12h[i-1] < -80 and  # was oversold
                price_above_ema and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R falls from overbought (> -20) with volume and below EMA100
            elif (williams_r_12h[i] < williams_r_12h[i-1] and  # falling from overbought
                  williams_r_12h[i-1] > -20 and  # was overbought
                  price_below_ema and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R reaches overbought OR price crosses below EMA100
            if (williams_r_12h[i] > -20) or (close[i] < ema100_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R reaches oversold OR price crosses above EMA100
            if (williams_r_12h[i] < -80) or (close[i] > ema100_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA100_Volume"
timeframe = "12h"
leverage = 1.0