#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA200 filter and volume confirmation.
# Enters long when Williams %R < -80 (oversold) and price > 1d EMA200 with volume spike.
# Enters short when Williams %R > -20 (overbought) and price < 1d EMA200 with volume spike.
# Exits when Williams %R crosses above -50 (long) or below -50 (short).
# Williams %R identifies overextended moves; EMA200 filters counter-trend trades.
# Volume spikes confirm institutional interest. Designed for low turnover in ranging/volatile markets.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # Calculate 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 12h timeframe
    williams_r_12h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 2.0 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for EMA200 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_12h[i]) or 
            np.isnan(ema200_12h[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma30[i])
        
        # Williams %R levels
        oversold = williams_r_12h[i] < -80
        overbought = williams_r_12h[i] > -20
        exit_long = williams_r_12h[i] > -50
        exit_short = williams_r_12h[i] < -50
        
        # Trend filter: price relative to 1d EMA200
        price_above_ema = close[i] > ema200_12h[i]
        price_below_ema = close[i] < ema200_12h[i]
        
        if position == 0:
            # Long: Oversold + above EMA200 + volume spike
            if (oversold and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Overbought + below EMA200 + volume spike
            elif (overbought and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0