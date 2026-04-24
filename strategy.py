#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA trend filter and volume confirmation.
- Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
- Long when Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.5 * 20-period average
- Short when Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.5 * 20-period average
- Exit when Williams %R reverses (%R > -50 for long, %R < -50 for short) OR volume drops below average
- Uses 1d primary with 1w HTF for EMA50 trend filter to avoid counter-trend trades
- Williams %R identifies momentum extremes; EMA50 filters for trend alignment; volume confirms conviction
- Designed to work in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 30-100 total trades over 4 years (7-25/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: price above/below 1w EMA50
    uptrend = close > ema_50_1w_aligned
    downtrend = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 20, 50)  # Need Williams %R, volume MA, and EMA data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume confirmation
            if williams_r[i] < -80 and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume confirmation
            elif williams_r[i] > -20 and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (recovering from oversold) OR volume drops
            if williams_r[i] > -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (declining from overbought) OR volume drops
            if williams_r[i] < -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0