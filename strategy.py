#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R Extreme with 1w EMA50 trend filter and volume confirmation.
- Williams %R: measures overbought/oversold levels (-20 oversold, -80 overbought)
- Long: Williams %R < -80 (oversold) + price > 1w EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: Williams %R > -20 (overbought) + price < 1w EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: Williams %R returns to -50 (mean reversion) OR opposite signal
- 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe to minimize fee drag
- Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets
- Williams %R is effective at catching reversals after extreme moves, suitable for 2025 bear/range conditions
"""

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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > 1w EMA50 (uptrend) + volume spike
            if volume_spike and williams_r[i] < -80 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 1w EMA50 (downtrend) + volume spike
            elif volume_spike and williams_r[i] > -20 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R >= -50 (mean reversion) OR price < 1w EMA50 (trend break)
            if williams_r[i] >= -50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R <= -50 (mean reversion) OR price > 1w EMA50 (trend break)
            if williams_r[i] <= -50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0