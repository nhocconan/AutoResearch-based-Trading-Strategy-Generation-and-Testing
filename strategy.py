#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion.
# In bull markets: buy oversold pullbacks in uptrend (price > 1d EMA50).
# In bear markets: sell overbought bounces in downtrend (price < 1d EMA50).
# Volume filter ensures participation. Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily Williams %R and EMA50 to 6h
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (moderate filter)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema50_6h[i]
        price_below_ema = close[i] < ema50_6h[i]
        
        # Williams %R levels: oversold < -80, overbought > -20
        oversold = williams_r_6h[i] < -80
        overbought = williams_r_6h[i] > -20
        
        if position == 0:
            # Long: Oversold + uptrend + volume
            if (oversold and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Overbought + downtrend + volume
            elif (overbought and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Overbought OR price breaks below EMA50
            if (overbought) or (close[i] < ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Oversold OR price breaks above EMA50
            if (oversold) or (close[i] > ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EMA50_Volume"
timeframe = "6h"
leverage = 1.0