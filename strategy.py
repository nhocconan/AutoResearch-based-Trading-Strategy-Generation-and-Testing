#!/usr/bin/env python3
# Hypothesis: 4h Williams %R Reversal with 1d trend filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold reversal) AND 1d close > 1d EMA34 (uptrend) AND 1d volume > 1.5 * 20-period average volume.
# Short when Williams %R crosses below -20 (overbought reversal) AND 1d close < 1d EMA34 (downtrend) AND 1d volume > 1.5 * 20-period average volume.
# Exit when Williams %R crosses below -50 for longs or above -50 for shorts.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 4h timeframe with strict entry conditions to avoid overtrading.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_WilliamsR_Reversal_1dEMA34_1dVolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)  # Volume > 1.5x 20-period MA
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Williams %R (14-period) on 4h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(williams_r[i-1])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold reversal) AND 1d close > 1d EMA34 (uptrend) AND volume confirmation
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought reversal) AND 1d close < 1d EMA34 (downtrend) AND volume confirmation
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 (momentum weakening)
            if williams_r[i-1] > -50 and williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 (momentum weakening)
            if williams_r[i-1] < -50 and williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals