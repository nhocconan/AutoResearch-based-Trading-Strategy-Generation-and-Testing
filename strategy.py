#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d trend filter and 6h volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold) AND 1d close > 1d EMA50 (bullish trend) AND 6h volume > 2.0 * 20-period average volume.
# Short when Williams %R crosses below -20 (overbought) AND 1d close < 1d EMA50 (bearish trend) AND 6h volume > 2.0 * 20-period average volume.
# Exit when Williams %R crosses below -50 (for longs) or above -50 (for shorts).
# Uses discrete position sizing (0.25) to limit fee churn. Williams %R is effective in ranging and trending markets, and the 1d EMA50 filter ensures we only trade with the higher timeframe trend, reducing whipsaws in bear markets like 2022 and 2025.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_WilliamsR_Reversal_1dEMA50_6hVolumeSpike_v1"
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
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume spike filter (LTF)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold) AND 1d EMA50 bullish trend AND volume spike
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought) AND 1d EMA50 bearish trend AND volume spike
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 (momentum weakening)
            if williams_r[i-1] >= -50 and williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 (momentum weakening)
            if williams_r[i-1] <= -50 and williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals