#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA200 Trend Filter
# - Williams %R(14) identifies overbought/oversold conditions
# - EMA200 on 1d determines primary trend (above = bullish, below = bearish)
# - Long when %R crosses above -80 (oversold bounce) AND price > 1d EMA200
# - Short when %R crosses below -20 (overbought rejection) AND price < 1d EMA200
# - Williams %R is effective in ranging markets; EMA200 filter avoids counter-trend trades
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Williams %R on 4h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R crossovers
        wr_cross_above_80 = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_cross_below_20 = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        price = close[i]
        ema200 = ema200_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 AND price above 1d EMA200
            if wr_cross_above_80 and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 AND price below 1d EMA200
            elif wr_cross_below_20 and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought) OR price falls below EMA200
            if wr_cross_below_20 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold) OR price rises above EMA200
            if wr_cross_above_80 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA200_TrendFilter"
timeframe = "4h"
leverage = 1.0