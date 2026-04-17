#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA50 and volume spike filter.
# Williams %R (14) measures momentum and identifies overbought/oversold conditions.
# Long: Williams %R crosses above -50 from below (bullish momentum) with price above 1d EMA50 and volume spike.
# Short: Williams %R crosses below -50 from above (bearish momentum) with price below 1d EMA50 and volume spike.
# Exit when Williams %R crosses back to -50 or price crosses 1d EMA50.
# This strategy aims to capture momentum shifts with strict filters to limit trades (target: 12-37/year).
# Works in bull markets (momentum continuation) and bear markets (mean reversion via oversold/overbought).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R (14) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align 1d EMA50 to 12h timeframe
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Williams %R conditions
        wr_above_50 = williams_r[i] > -50
        wr_below_50 = williams_r[i] < -50
        wr_crossed_above_50 = wr_above_50 and (i == start_idx or williams_r[i-1] <= -50)
        wr_crossed_below_50 = wr_below_50 and (i == start_idx or williams_r[i-1] >= -50)
        
        # Price relative to 1d EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        if position == 0:
            # Long: Williams %R crosses above -50 with price above EMA and volume spike
            if (wr_crossed_above_50 and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50 with price below EMA and volume spike
            elif (wr_crossed_below_50 and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses below -50 OR price crosses below EMA
            if (wr_crossed_below_50 or price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses above -50 OR price crosses above EMA
            if (wr_crossed_above_50 or price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0