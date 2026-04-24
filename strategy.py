#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
- Long when Williams %R crosses above -80 (oversold reversal) AND 12h close > 12h EMA50 (bullish trend) AND volume spike
- Short when Williams %R crosses below -20 (overbought reversal) AND 12h close < 12h EMA50 (bearish trend) AND volume spike
- Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short)
- Uses 4h primary with 12h HTF to target 75-200 total trades over 4 years (19-50/year)
- Williams %R captures momentum reversals; EMA50 filters trend; volume spike confirms strength
- Designed for range-bound and trending markets, works in both bull and bear regimes
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Williams %R (14-period) on 4h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Get 12h HTF data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_12h_aligned
    bearish_regime = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND bullish regime AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND bearish regime AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought)
            if williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold)
            if williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0