#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversal with 1d EMA34 Trend Filter and Volume Spike.
- Williams %R(14) identifies oversold (< -80) and overbought (> -20) conditions on 12h.
- Extreme readings (> -10 for long, < -90 for short) with volume spike (> 2.0x 24-period avg) capture exhaustion moves.
- 1d EMA34 provides higher-timeframe trend filter to avoid counter-trend trades.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag.
- Works in bull/bear markets via 1d trend filter and volatility-based logic.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 12h
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    # Avoid division by zero
    rr_denom = highest_high - lowest_low
    williams_r = np.where(rr_denom != 0, -100 * (highest_high - close) / rr_denom, -50)
    
    # Volume confirmation: > 2.0x 24-period average (strict for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
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
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: Williams %R > -10 (extreme oversold) + above 1d EMA34 (bullish higher-timeframe trend)
                if williams_r[i] > -10 and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R < -90 (extreme overbought) + below 1d EMA34 (bearish higher-timeframe trend)
                elif williams_r[i] < -90 and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R < -50 (reversal from oversold) OR below EMA34 (trend change)
            if williams_r[i] < -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R > -50 (reversal from overbought) OR above EMA34 (trend change)
            if williams_r[i] > -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0