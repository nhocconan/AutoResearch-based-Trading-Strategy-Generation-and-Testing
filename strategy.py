#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme with 1d EMA34 Trend Filter and Volume Confirmation.
- Williams %R identifies overbought/oversold conditions; extremes (< -80 or > -20) signal potential reversals.
- 1d EMA34 provides higher-timeframe trend filter to align with intermediate momentum and reduce counter-trend trades.
- Volume confirmation (>1.5x 24-period average) validates the reversal signal and reduces false entries.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 50-150 total over 4 years (12-37/year) on 12h timeframe to avoid fee drag.
- Works in bull/bear markets via 1d trend filter and volatility-based volume confirmation.
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.5x 24-period average volume (12h * 2 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, period) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike and above 1d EMA34 (bullish higher-timeframe trend)
            if williams_r[i] < -80 and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume spike and below 1d EMA34 (bearish higher-timeframe trend)
            elif williams_r[i] > -20 and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -50 (momentum weakening) OR price closes below 1d EMA34 (trend change)
            if williams_r[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -50 (momentum weakening) OR price closes above 1d EMA34 (trend change)
            if williams_r[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0