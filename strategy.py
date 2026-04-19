#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w trend filter, volume confirmation, and ATR stop.
# Uses daily Bollinger Bands (20, 2) for breakout entries in the direction of the weekly trend.
# Weekly trend defined by EMA(34) slope to avoid whipsaw.
# Volume filter ensures breakouts have conviction (volume > 1.5x 20-day average).
# Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull/bear markets.
name = "1d_Bollinger_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper_band = (basis + 2 * dev).values
    lower_band = (basis - 2 * dev).values
    
    # Weekly trend filter: EMA(34) slope
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema34_slope = np.diff(ema34_1w_aligned, prepend=ema34_1w_aligned[0])
    
    # Volume filter: volume > 1.5 * 20-day average
    volume_ma = close_series.rolling(window=20, min_periods=20).mean()
    volume_ma_vals = volume_ma.values
    volume_filter = volume > (volume_ma_vals * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_slope[i]) or
            np.isnan(volume_ma_vals[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper band + weekly uptrend (positive slope) + volume filter
            if (close[i] > upper_band[i] and 
                ema34_slope[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + weekly downtrend (negative slope) + volume filter
            elif (close[i] < lower_band[i] and 
                  ema34_slope[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below basis OR weekly trend turns down
            if (close[i] < basis[i]) or (ema34_slope[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above basis OR weekly trend turns up
            if (close[i] > basis[i]) or (ema34_slope[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals