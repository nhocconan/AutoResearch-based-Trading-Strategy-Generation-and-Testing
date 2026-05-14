#!/usr/bin/env python3
# Hypothesis: 4h Williams %R reversal with 1d EMA trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) AND 1d EMA50 > EMA200 (bullish trend) AND 1d volume > 2.0 * 20-period average volume.
# Short when Williams %R crosses below -20 (overbought reversal) AND 1d EMA50 < EMA200 (bearish trend) AND 1d volume > 2.0 * 20-period average volume.
# Exit when Williams %R returns to -50 (mean reversion) or opposing signal occurs.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with strict entry conditions.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h.

name = "4h_WilliamsR_Reversal_1dEMA_Trend_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # Calculate 1d EMA50 and EMA200 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend = align_htf_to_ltf(prices, df_1d, ema_50 > ema_200)  # Boolean: True for bullish, False for bearish
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Williams %R (14-period) on 4h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_trend[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(williams_r[i-1])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold reversal) AND bullish trend AND volume confirmation
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                ema_trend[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought reversal) AND bearish trend AND volume confirmation
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  not ema_trend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) or short signal occurs
            if williams_r[i] >= -50:  # Return to mean level
                signals[i] = 0.0
                position = 0
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and  # Opposing short signal
                  not ema_trend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) or long signal occurs
            if williams_r[i] <= -50:  # Return to mean level
                signals[i] = 0.0
                position = 0
            elif (williams_r[i-1] <= -80 and williams_r[i] > -80 and  # Opposing long signal
                  ema_trend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals