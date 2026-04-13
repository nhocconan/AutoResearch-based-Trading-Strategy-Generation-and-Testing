# 6h_1d_Liquidity_Pooler
# Hypothesis: Uses daily liquidity pools (recent highs/lows) as support/resistance on 6h chart.
# In bull markets: price sweeps liquidity pools then continues trend.
# In bear markets: price sweeps liquidity pools then reverses.
# Uses volume confirmation to distinguish real breaks from fakeouts.
# Target: 20-40 trades/year (80-160 total over 4 years) on 6h timeframe.
# Works in both bull and bear by trading liquidity sweeps with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for liquidity pools (recent highs/lows)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate recent daily highs and lows (liquidity pools)
    # Use 10-day lookback for recent swing highs/lows
    recent_high_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max()
    recent_low_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min()
    
    # Add small buffer to avoid exact equality issues
    liquidity_high = recent_high_1d * 1.001  # 0.1% above recent high
    liquidity_low = recent_low_1d * 0.999    # 0.1% below recent low
    
    # Align liquidity levels to 6h timeframe
    liquidity_high_aligned = align_htf_to_ltf(prices, df_1d, liquidity_high.values)
    liquidity_low_aligned = align_htf_to_ltf(prices, df_1d, liquidity_low.values)
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume must be 2x average
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(liquidity_high_aligned[i]) or 
            np.isnan(liquidity_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above liquidity high with volume spike
        if (close[i] > liquidity_high_aligned[i] and 
            volume_spike[i] and 
            position != 1):
            position = 1
            signals[i] = position_size
        # Short entry: price breaks below liquidity low with volume spike
        elif (close[i] < liquidity_low_aligned[i] and 
              volume_spike[i] and 
              position != -1):
            position = -1
            signals[i] = -position_size
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_Liquidity_Pooler"
timeframe = "6h"
leverage = 1.0