#!/usr/bin/env python3
# Hypothesis: 6h timeframe with 1-day RSI and Bollinger Band squeeze for mean reversion.
# In low volatility regimes (BB width < 30th percentile), price tends to mean-revert.
# Enters long when RSI(14) < 30 and price > Bollinger lower band, short when RSI(14) > 70 and price < Bollinger upper band.
# Uses 1-day RSI as momentum filter: only take longs when 1d RSI > 50, shorts when < 50.
# Exits when volatility regime shifts to high volatility or RSI crosses 50.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_RSI_BB_Squeeze_MeanReversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    sma_20 = close_1d.rolling(window=20, min_periods=20).mean()
    std_20 = close_1d.rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: low volatility when BB width < 30th percentile
    bb_width_percentile = bb_width.rolling(window=50, min_periods=50).quantile(0.3)
    bb_squeeze = bb_width < bb_width_percentile
    bb_squeeze_values = bb_squeeze.values
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze_values)
    
    # Bollinger Bands for entry/exit
    upper_bb_values = upper_bb.values
    lower_bb_values = lower_bb.values
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_values)
    
    # 1-day RSI(14) for momentum filter
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # 6h RSI(14) for entry signals
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.where(delta_6h > 0, 0)
    loss_6h = -delta_6h.where(delta_6h < 0, 0)
    avg_gain_6h = gain_6h.rolling(window=14, min_periods=14).mean()
    avg_loss_6h = loss_6h.rolling(window=14, min_periods=14).mean()
    rs_6h = avg_gain_6h / avg_loss_6h
    rsi_6h = 100 - (100 / (1 + rs_6h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_squeeze_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility + RSI oversold + price above lower BB + 1d RSI bullish
            if (bb_squeeze_aligned[i] and 
                rsi_6h[i] < 30 and 
                close[i] > lower_bb_aligned[i] and 
                rsi_1d_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility + RSI overbought + price below upper BB + 1d RSI bearish
            elif (bb_squeeze_aligned[i] and 
                  rsi_6h[i] > 70 and 
                  close[i] < upper_bb_aligned[i] and 
                  rsi_1d_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR RSI crosses 50
            if (not bb_squeeze_aligned[i]) or (rsi_6h[i] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR RSI crosses 50
            if (not bb_squeeze_aligned[i]) or (rsi_6h[i] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals