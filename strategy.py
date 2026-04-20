#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) identifies overbought/oversold conditions
# - 1d EMA(50) determines trend direction (only trade in trend direction)
# - Volume > 1.3x 20-period average confirms momentum
# - Williams %R extremes: < -80 for long, > -20 for short
# - Exit when Williams %R crosses -50 (mean reversion midpoint)
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 4h price and volume data
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema_50 = ema_50_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above EMA50 (uptrend) + volume surge
            if wr < -80 and price > ema_50 and vol > 1.3 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below EMA50 (downtrend) + volume surge
            elif wr > -20 and price < ema_50 and vol > 1.3 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_EMA50_Volume"
timeframe = "4h"
leverage = 1.0