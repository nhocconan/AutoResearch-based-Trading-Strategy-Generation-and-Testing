#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) on 4h: oversold < -80 for long, overbought > -20 for short
# - Trend filter: 1d EMA50 - price must be above EMA50 for long, below for short
# - Volume confirmation: current volume > 1.5x 20-period average
# - Exit: Williams %R crosses back above -50 (long) or below -50 (short)
# - Position size: 0.25 to limit drawdown
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data for Williams %R and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above 1d EMA50 + volume surge
            if wr < -80 and price > ema_50_1d_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below 1d EMA50 + volume surge
            elif wr > -20 and price < ema_50_1d_aligned[i] and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back above -50
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back below -50
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dTrendFilter_Volume"
timeframe = "4h"
leverage = 1.0