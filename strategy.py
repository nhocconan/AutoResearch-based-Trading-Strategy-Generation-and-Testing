#!/usr/bin/env python3
"""
6H_Stochastic_Bollinger_Squeeze_1dTrend
Hypothesis: Bollinger Band squeeze (low volatility) on 6h combined with Stochastic oversold/overbought conditions,
filtered by 1d trend direction. This captures mean-reversion bounces in low-volatility environments,
which occur in both bull and bear markets. Trend filter prevents counter-trend trades.
Target: 50-150 trades over 4 years (12-37/year).
"""

name = "6H_Stochastic_Bollinger_Squeeze_1dTrend"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_std_dev)
    lower_band = sma - (bb_std * bb_std_dev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band squeeze: width < 20-period average width * 0.5
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < (bb_width_ma * 0.5)
    
    # Stochastic Oscillator (14,3,3) on 6h
    stoch_k_period = 14
    stoch_d_period = 3
    lowest_low = pd.Series(low).rolling(window=stoch_k_period, min_periods=stoch_k_period).min().values
    highest_high = pd.Series(high).rolling(window=stoch_k_period, min_periods=stoch_k_period).max().values
    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stoch_k = np.where((highest_high - lowest_low) == 0, 50, stoch_k)  # avoid div by zero
    stoch_d = pd.Series(stoch_k).rolling(window=stoch_d_period, min_periods=stoch_d_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, stoch_k_period) + 20  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(sma[i]) or np.isnan(bb_width[i]) or \
           np.isnan(bb_width_ma[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long: squeeze + Stochastic oversold (K < 20) + uptrend
            if (squeeze_condition[i] and 
                stoch_k[i] < 20 and 
                stoch_d[i] < 20 and
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: squeeze + Stochastic overbought (K > 80) + downtrend
            elif (squeeze_condition[i] and 
                  stoch_k[i] > 80 and 
                  stoch_d[i] > 80 and
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Stochastic overbought (K > 80) or squeeze ends
            if (stoch_k[i] > 80 or not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Stochastic oversold (K < 20) or squeeze ends
            if (stoch_k[i] < 20 or not squeeze_condition[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals