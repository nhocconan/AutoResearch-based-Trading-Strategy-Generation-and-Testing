#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND close > 1d EMA50 (uptrend) AND volume > 1.5x 20-period MA.
Short when Williams %R > -20 (overbought) AND close < 1d EMA50 (downtrend) AND volume > 1.5x 20-period MA.
Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is hit.
Designed to capture reversals in ranging markets while respecting higher timeframe trend.
Target: 20-30 trades/year per symbol to avoid fee drag.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 4h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # need EMA50, Williams %R, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        mean_reversion = abs(williams_r[i] + 50) < 5  # Within 5 points of -50
        opposite_extreme = (position == 1 and overbought) or \
                           (position == -1 and oversold)
        
        if position == 0:
            # Long: Oversold AND uptrend AND volume confirmation
            if oversold and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND downtrend AND volume confirmation
            elif overbought and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: mean reversion or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = mean_reversion or opposite_extreme
            elif position == -1:
                exit_signal = mean_reversion or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0