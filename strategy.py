#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day Bollinger Band Filter
# Williams %R identifies overbought/oversold conditions on 6h chart.
# Bollinger Bands on 1-day chart define volatility regime: trade only when price is outside bands (high volatility breakouts).
# Long when Williams %R < -80 (oversold) and price > upper BB(1d)
# Short when Williams %R > -20 (overbought) and price < lower BB(1d)
# Uses Bollinger Band squeeze/expansion as volatility filter to avoid ranging markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.
# Works in both bull and bear markets by capturing volatility breakouts from extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    
    # Williams %R on 6h (14 periods)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup for BB
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            continue
        
        # Long: Williams %R oversold (< -80) and price above 1-day upper BB (volatility breakout up)
        if (williams_r[i] < -80 and 
            close[i] > upper_bb_aligned[i]):
            signals[i] = 0.25
        
        # Short: Williams %R overbought (> -20) and price below 1-day lower BB (volatility breakout down)
        elif (williams_r[i] > -20 and 
              close[i] < lower_bb_aligned[i]):
            signals[i] = -0.25
        
        # Exit: Williams %R returns to neutral range (-50 to -50) or opposite extreme
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and williams_r[i] > -50) or
               (signals[i-1] == -0.25 and williams_r[i] < -50))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WilliamsR_1dBB_VolatilityBreakout"
timeframe = "6h"
leverage = 1.0