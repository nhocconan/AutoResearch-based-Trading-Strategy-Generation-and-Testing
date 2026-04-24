#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R (14) with 1d EMA50 trend filter and volume confirmation.
- Long when Williams %R crosses above -80 (oversold reversal) AND close > 1d EMA50 (bullish trend) AND volume > 1.5 * median volume of last 20 bars
- Short when Williams %R crosses below -20 (overbought reversal) AND close < 1d EMA50 (bearish trend) AND volume > 1.5 * median volume of last 20 bars
- Exit on opposite Williams %R cross or trend reversal (close crosses 1d EMA50)
- Uses 6h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Williams %R captures mean reversion in ranging markets while respecting higher timeframe trend
- Volume confirmation reduces false signals from low-participation moves
- Designed for BTC/ETH with edge in both trending (pullback entries) and ranging (extreme reversals) markets
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
    
    # Calculate Williams %R (14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below), trend up, volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above), trend down, volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR trend reversal (close < EMA50)
            if williams_r[i] > -20 and williams_r[i-1] <= -20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR trend reversal (close > EMA50)
            if williams_r[i] < -80 and williams_r[i-1] >= -80 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0