#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R extreme with 1w EMA50 trend filter and volume spike.
- Primary timeframe: 1d for execution and signal generation.
- HTF: 1w for EMA50 trend filter (bullish above EMA50, bearish below).
- Williams %R calculated on 1d: long when %R < -80 (oversold) with volume spike and price > 1w EMA50.
                            short when %R > -20 (overbought) with volume spike and price < 1w EMA50.
- Exit: When Williams %R returns to -50 (mean reversion edge) or opposite extreme triggers flip.
- Works in bull by buying oversold dips in uptrend, in bear by selling overbought rallies in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, lookback=14):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = wr.fillna(-50)  # Neutral value when range is zero
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Williams %R on 1d
    wr_1d = calculate_williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough 1w bars for EMA50, volume MA, and WR lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(wr_1d[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for extreme Williams %R signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish: Williams %R < -80 (oversold) and price > 1w EMA50 (uptrend)
                if wr_1d[i] < -80 and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R > -20 (overbought) and price < 1w EMA50 (downtrend)
                elif wr_1d[i] > -20 and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) or flipped to overbought
            if wr_1d[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) or flipped to oversold
            if wr_1d[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0