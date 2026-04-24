#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R (14) with 1d EMA(50) trend filter and volume spike confirmation.
- Williams %R > -20 = overbought (short signal), < -80 = oversold (long signal).
- Only take signals aligned with 1d EMA(50) trend: long if close > EMA50, short if close < EMA50.
- Volume confirmation: current 6h volume > 1.5 * 20-period volume MA to avoid false signals.
- Exits: Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
- Discrete signal size: 0.25 to manage drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull/bear: mean reversion in ranges, trend-filtered to avoid counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R (14-period) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough for 1d EMA, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if curr_vol_spike:
                # Long: oversold (-80 or below) and above 1d EMA50 (uptrend)
                if curr_wr <= -80 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought (-20 or above) and below 1d EMA50 (downtrend)
                elif curr_wr >= -20 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back above -50 (momentum fading)
            if curr_wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back below -50 (momentum fading)
            if curr_wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0