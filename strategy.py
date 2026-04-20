#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA Trend Filter
# - Williams %R(14) on 12h for momentum reversal signals
# - Long when %R < -80 (oversold) and 1d EMA(34) > 1d EMA(89) (uptrend bias)
# - Short when %R > -20 (overbought) and 1d EMA(34) < 1d EMA(89) (downtrend bias)
# - Williams %R captures short-term reversals; dual EMA filters for intermediate trend direction
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) and EMA(89) on 1d timeframe
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_89_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        wr = williams_r[i]
        ema_fast = ema_34_1d_aligned[i]
        ema_slow = ema_89_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + EMA(34) > EMA(89) (uptrend)
            if wr < -80 and ema_fast > ema_slow:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + EMA(34) < EMA(89) (downtrend)
            elif wr > -20 and ema_fast < ema_slow:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or EMA trend turns bearish
            if wr > -50 or ema_fast < ema_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or EMA trend turns bullish
            if wr < -50 or ema_fast > ema_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA_TrendFilter"
timeframe = "12h"
leverage = 1.0