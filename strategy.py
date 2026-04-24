#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h EMA50 trend filter and volume spike confirmation.
- Long when Williams %R crosses above -80 (oversold reversal) AND close > 12h EMA50 (bullish trend) AND volume > 1.5 * median volume of last 20 bars
- Short when Williams %R crosses below -20 (overbought reversal) AND close < 12h EMA50 (bearish trend) AND volume > 1.5 * median volume of last 20 bars
- Exit on opposite Williams %R cross or trend reversal (close crosses 12h EMA50)
- Uses 6h primary timeframe with 12h HTF to target 50-150 total trades over 4 years (12-37/year)
- Williams %R identifies momentum exhaustion points that work in ranging markets (mean reversion at extremes)
- 12h EMA50 ensures alignment with higher timeframe trend to avoid whipsaws
- Volume spike filter adapts to changing market conditions, reducing false breakouts
- Designed for BTC/ETH with edge in both trending and ranging markets
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
    
    # Calculate Williams %R (14 period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below), trend up (close > EMA50), volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above), trend down (close < EMA50), volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema_50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (from above) OR trend reversal (close < EMA50)
            if williams_r[i] < -20 and williams_r[i-1] >= -20 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (from below) OR trend reversal (close > EMA50)
            if williams_r[i] > -80 and williams_r[i-1] <= -80 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0