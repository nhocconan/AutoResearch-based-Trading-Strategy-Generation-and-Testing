#!/usr/bin/env python3
"""
6h_1W_1D_Volatility_Squeeze_Breakout
Hypothesis: In 6h timeframe, buy when price breaks above weekly Bollinger upper band with volatility contraction (BBW < 50th percentile) and 1d EMA20 alignment, sell when breaks below lower band with same conditions. Uses weekly volatility regime to filter breakouts, works in bull (breakouts continuation) and bear (mean reversion in squeeze) markets. Targets 15-35 trades/year with 30% position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for Bollinger Bands and volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20, 2)
    weekly_close = df_1w['close'].values
    bb_period = 20
    bb_std = 2
    sma = pd.Series(weekly_close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(weekly_close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Bollinger Width for volatility regime
    bb_width = (upper_band - lower_band) / sma
    # Historical 50th percentile of BB width (volatility median)
    bb_width_median = pd.Series(bb_width).rolling(window=50, min_periods=50).median().values
    volatility_squeeze = bb_width < bb_width_median  # True when volatility below median
    
    # Align weekly indicators to 6h
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    volatility_squeeze_aligned = align_htf_to_ltf(prices, df_1w, volatility_squeeze.astype(float))
    
    # Daily EMA20 for trend alignment
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.30  # 30% position size
    
    for i in range(100, n):
        # Skip if any data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(volatility_squeeze_aligned[i]) or np.isnan(ema20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above weekly upper band + volatility squeeze + price > daily EMA20
        long_signal = (close[i] > upper_band_aligned[i] and 
                      volatility_squeeze_aligned[i] > 0.5 and  # True when aligned
                      close[i] > ema20_aligned[i])
        
        # Short: price breaks below weekly lower band + volatility squeeze + price < daily EMA20
        short_signal = (close[i] < lower_band_aligned[i] and 
                       volatility_squeeze_aligned[i] > 0.5 and  # True when aligned
                       close[i] < ema20_aligned[i])
        
        if position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below weekly lower band OR price < daily EMA20
            if close[i] < lower_band_aligned[i] or close[i] < ema20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above weekly upper band OR price > daily EMA20
            if close[i] > upper_band_aligned[i] or close[i] > ema20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1W_1D_Volatility_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0