#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R combined with 1d EMA50 trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions (below -80 = oversold, above -20 = overbought).
# In trending markets, we look for reversals from extreme levels back toward the mean.
# Uses 1d EMA50 for trend direction and volume spike for confirmation.
# Designed to capture mean reversion within trends with low turnover.
# Target: 12-30 trades/year to stay within optimal range for 6h timeframe.

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
    
    # Calculate 14-period Williams %R on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Williams %R and EMA to 6h
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 14-period Williams + EMA50 + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_6h[i]) or 
            np.isnan(ema50_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average (moderate to balance signal quality)
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Williams %R levels
        oversold = williams_r_6h[i] < -80
        overbought = williams_r_6h[i] > -20
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_6h[i]
        price_below_ema = close[i] < ema50_6h[i]
        
        if position == 0:
            # Long: Williams %R oversold AND price above EMA50 (bullish reversal in uptrend)
            if (oversold and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price below EMA50 (bearish reversal in downtrend)
            elif (overbought and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral territory (> -50) OR price crosses below EMA50
            if (williams_r_6h[i] > -50) or (close[i] < ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral territory (< -50) OR price crosses above EMA50
            if (williams_r_6h[i] < -50) or (close[i] > ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0