#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; entries taken when
# extreme readings reverse in direction of higher timeframe trend. Designed
# to capture mean reversion within trends with low turnover. Target: 15-30
# trades/year to stay within optimal range for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R on 1d
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Calculate 50-period EMA on 1d
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h
    williams_r_12h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 14-period Williams %R + EMA50 + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Williams %R conditions
        wr_oversold = williams_r_12h[i] <= -80  # Oversold threshold
        wr_overbought = williams_r_12h[i] >= -20  # Overbought threshold
        wr_crossing_up = (williams_r_12h[i] > williams_r_12h[i-1])  # Turning up from oversold
        wr_crossing_down = (williams_r_12h[i] < williams_r_12h[i-1])  # Turning down from overbought
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        if position == 0:
            # Long: Williams %R crosses up from oversold with volume and price above EMA
            if (wr_oversold and wr_crossing_up and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses down from overbought with volume and price below EMA
            elif (wr_overbought and wr_crossing_down and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R reaches overbought OR price crosses below EMA
            if (williams_r_12h[i] >= -20) or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R reaches oversold OR price crosses above EMA
            if (williams_r_12h[i] <= -80) or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0