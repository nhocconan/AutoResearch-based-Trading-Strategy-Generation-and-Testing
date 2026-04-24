#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1w EMA trend filter and volume confirmation.
- Williams %R(14) measures overbought/oversold levels: Long when %R < -80, Short when %R > -20
- 1w EMA50 filter ensures we trade with the weekly trend: Long only when price > 1w EMA50, Short only when price < 1w EMA50
- Volume confirmation: current volume > 1.5 * 20-period average to ensure conviction
- Exit when Williams %R reverts to neutral territory (-50) or trend weakens
- Designed to capture mean reversion in strong trends, avoiding counter-trend trades
- Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets
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
    
    # Calculate Williams %R(14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if price > 1w EMA50, bearish if price < 1w EMA50
    bullish_trend = close > ema_50_1w_aligned
    bearish_trend = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 50) + 1  # Need Williams %R, volume MA, and EMA data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish trend AND volume confirmation
            if williams_r[i] < -80 and bullish_trend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND bearish trend AND volume confirmation
            elif williams_r[i] > -20 and bearish_trend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R >= -50 (reverted to neutral) OR trend turns bearish
            if williams_r[i] >= -50 or not bullish_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R <= -50 (reverted to neutral) OR trend turns bullish
            if williams_r[i] <= -50 or not bearish_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0