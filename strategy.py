#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy with 1d Williams %R momentum and 1w EMA trend filter
# Williams %R < -80 indicates oversold, > -20 indicates overbought
# 1w EMA filter ensures trades align with higher timeframe trend
# Williams %R mean reversion within trend has proven edge in BTC/ETH
# Target: 20-40 trades/year with controlled risk

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14 periods)
    wr_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high_1d).rolling(window=wr_len, min_periods=wr_len).max().values
    lowest_low = pd.Series(low_1d).rolling(window=wr_len, min_periods=wr_len).min().values
    
    # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # Avoid division by zero
    
    # Align Williams %R to 4h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (21 periods)
    ema_len = 21
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # Align EMA to 4h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, wr_len, ema_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price above/below 1w EMA
        above_ema = price > ema_1w_aligned[i]
        below_ema = price < ema_1w_aligned[i]
        
        # Williams %R signals
        oversold = wr_aligned[i] < -80
        overbought = wr_aligned[i] > -20
        
        if position == 0:
            # Enter long: uptrend + oversold
            if above_ema and oversold:
                position = 1
                signals[i] = position_size
            # Enter short: downtrend + overbought
            elif below_ema and overbought:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (momentum shift) OR trend change
            if wr_aligned[i] > -50 or not above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (momentum shift) OR trend change
            if wr_aligned[i] < -50 or not below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dWR_1wEMA_Trend_Momentum_v1"
timeframe = "4h"
leverage = 1.0