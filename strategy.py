# 1. State your hypothesis:
# The strategy uses 4-hour timeframe with a focus on capturing breakouts from key
# support/resistance levels derived from the previous daily candlestick (Camarilla levels).
# It combines:
#   - A trend filter using the 12-hour EMA (50-period) to ensure trades are taken
#     in the direction of the higher timeframe trend.
#   - Volume confirmation requiring the current bar's volume to exceed 2.5 times its
#     20-period moving average, ensuring breakouts are backed by strong participation.
#   - Discrete position sizing of 0.25 to balance return potential and drawdown control.
# This approach aims to reduce trade frequency (targeting ~20-30 trades/year) to minimize
# fee drag while maintaining robustness across both bull and bear markets by relying on
# institutional levels (Camarilla) and trend alignment.
#
# 2. Implementation:
#   - Uses Camarilla R4 and S4 levels from the prior daily bar (avoiding look-ahead via shift).
#   - Aligns these levels and the 12h EMA50 to the 4h chart using the provided mtf_data helpers.
#   - Applies volume confirmation and trend filter as entry conditions.
#   - Exits when the trend fails or price retraces back through the entry level.
#   - All calculations use proper min_periods to avoid look-ahead.
#
# 3. Trade frequency control:
#   - Strict volume threshold (2.5x average) and trend requirement limit entries.
#   - Position size is fixed at ±0.25 to reduce churn from small adjustments.
#   - Target is well below the 400-trade hard limit for 4h strategies.

#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike_v3
Hypothesis: Focus on high-probability breakouts at institutional Camarilla levels (R4/S4)
with 12h EMA50 trend filter and strict volume confirmation (>2.5x average) to reduce
trade frequency and improve robustness across bull/bear markets. Position size set to 0.25.
Target: 20-30 trades/year to avoid fee drag.
"""

name = "4h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R4 and S4 levels
    rng = prev_high - prev_low
    r4 = prev_close + (rng * 1.1 / 2)  # R4 = C + (H-L) * 1.1/2
    s4 = prev_close - (rng * 1.1 / 2)  # S4 = C - (H-L) * 1.1/2
    
    # Align 1d levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_12h = close[i] > ema_50_12h_aligned[i]
        downtrend_12h = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>2.5x average volume - stricter)
        volume_confirm = volume[i] > volume_ma[i] * 2.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above R4 + volume confirmation
            if uptrend_12h and close[i] > r4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below S4 + volume confirmation
            elif downtrend_12h and close[i] < s4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R4
            if not uptrend_12h or close[i] < r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S4
            if not downtrend_12h or close[i] > s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals