#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme reversal with 1w EMA trend filter and 1d volume confirmation.
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 1w EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 1w EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to balance return and drawdown. Designed for 6h timeframe to capture reversals in both trending and ranging markets with proper risk control.
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe.

name = "6h_WilliamsR_Reversal_1wEMA50_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate Williams %R on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)  # Volume > 1.5x 20-period average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold) AND price > 1w EMA50 AND volume spike
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought) AND price < 1w EMA50 AND volume spike
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) or reaches overbought
            if williams_r[i] >= -50 or williams_r[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) or reaches oversold
            if williams_r[i] <= -50 or williams_r[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals