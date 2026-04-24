#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme + 1d EMA50 trend filter with volume spike confirmation.
- Long when Williams %R < -80 (oversold) AND close > 1d EMA50 (bullish trend) AND volume > 2.0 * median volume of last 20 bars
- Short when Williams %R > -20 (overbought) AND close < 1d EMA50 (bearish trend) AND volume > 2.0 * median volume of last 20 bars
- Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR trend reverses (close crosses 1d EMA50)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Williams %R identifies exhaustion points in both trending and ranging markets
- 1d EMA50 ensures alignment with higher timeframe trend to avoid whipsaws
- Volume spike confirmation reduces false signals from low-volatility environments
- Designed for BTC/ETH with edge in mean reversion during extremes while respecting higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80), trend up (close > EMA50), volume confirmation
            if williams_r[i] < -80 and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), trend down (close < EMA50), volume confirmation
            elif williams_r[i] > -20 and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 OR trend reversal (close < EMA50)
            if williams_r[i] > -50 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 OR trend reversal (close > EMA50)
            if williams_r[i] < -50 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0