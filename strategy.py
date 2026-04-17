#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Supertrend(ATR=10, mult=3) with 1d volume confirmation and 1w EMA200 trend filter.
# Supertrend adapts to volatility and captures trends while avoiding whipsaws in ranging markets.
# Volume confirmation ensures breakouts have institutional participation.
# Weekly EMA200 filter ensures alignment with long-term trend, improving performance in both bull and bear markets.
# Designed for low turnover (target: 12-37 trades/year) with clear entry/exit rules.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate ATR(10) on 1d
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend components on 1d
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    start_idx = max(10, 1)  # Need at least 10 periods for ATR
    if start_idx < len(close_1d):
        supertrend[start_idx] = upper_band[start_idx]
        direction[start_idx] = 1
    
    # Calculate Supertrend iteratively
    for i in range(start_idx + 1, len(close_1d)):
        if np.isnan(supertrend[i-1]) or np.isnan(direction[i-1]):
            supertrend[i] = upper_band[i]
            direction[i] = 1
            continue
            
        if close_1d[i] <= supertrend[i-1]:
            # Downtrend
            supertrend[i] = max(upper_band[i], supertrend[i-1])
            direction[i] = -1
        else:
            # Uptrend
            supertrend[i] = min(lower_band[i], supertrend[i-1])
            direction[i] = 1
    
    # Calculate 1d volume MA20 for confirmation
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA200 for long-term trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to 12h timeframe
    supertrend_12h = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_12h = align_htf_to_ltf(prices, df_1d, direction)
    volume_ma20_12h = align_htf_to_ltf(prices, df_1d, volume_ma20)
    ema200_1w_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_12h[i]) or 
            np.isnan(direction_12h[i]) or 
            np.isnan(volume_ma20_12h[i]) or 
            np.isnan(ema200_1w_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average (moderate to balance signal quality)
        volume_filter = volume[i] > (1.5 * volume_ma20_12h[i])
        
        # Trend filter: price above/below 1w EMA200
        price_above_ewma = close[i] > ema200_1w_12h[i]
        price_below_ewma = close[i] < ema200_1w_12h[i]
        
        if position == 0:
            # Long: Supertrend uptrend (direction=1) with volume confirmation and above weekly EMA200
            if (direction_12h[i] == 1 and volume_filter and price_above_ewma):
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend (direction=-1) with volume confirmation and below weekly EMA200
            elif (direction_12h[i] == -1 and volume_filter and price_below_ewma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns down OR price crosses below weekly EMA200
            if (direction_12h[i] == -1) or (close[i] < ema200_1w_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns up OR price crosses above weekly EMA200
            if (direction_12h[i] == 1) or (close[i] > ema200_1w_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Supertrend_ATR10_mult3_1dVolume_1wEMA200"
timeframe = "12h"
leverage = 1.0