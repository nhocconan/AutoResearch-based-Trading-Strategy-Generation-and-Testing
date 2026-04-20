#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA Trend Filter
# - Williams %R (14) on 6h identifies overbought/oversold conditions
# - Long when %R < -80 (oversold) and 1d EMA(50) > 1d EMA(200) (uptrend)
# - Short when %R > -20 (overbought) and 1d EMA(50) < 1d EMA(200) (downtrend)
# - Williams %R captures short-term reversals; dual EMA filters for intermediate trend direction
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) and EMA(200) on 1d timeframe
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate Williams %R (14) on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after EMA(200) warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        wr = williams_r[i]
        ema50 = ema_50_aligned[i]
        ema200 = ema_200_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + EMA50 > EMA200 (uptrend)
            if wr < -80 and ema50 > ema200:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + EMA50 < EMA200 (downtrend)
            elif wr > -20 and ema50 < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or EMA50 < EMA200
            if wr > -50 or ema50 < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or EMA50 > EMA200
            if wr < -50 or ema50 > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA_TrendFilter"
timeframe = "6h"
leverage = 1.0