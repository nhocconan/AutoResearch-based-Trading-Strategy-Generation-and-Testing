#!/usr/bin/env python3
# Hypothesis: 4h Williams %R reversal with 1d EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below AND price > 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Short when Williams %R crosses below -20 from above AND price < 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Exit on opposite Williams %R cross (long exit when %R crosses below -20, short exit when %R crosses above -80).
# Uses discrete position sizing (0.25) to limit fee churn. Williams %R identifies overbought/oversold levels with momentum,
# effective in both bull and bear markets when combined with trend and volume filters.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_WilliamsR_Reversal_1dEMA50_1dVolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.5 * vol_ma_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    # Calculate Williams %R (14-period) on 4h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(williams_r[i-1])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND price > 1d EMA50 AND volume confirmation
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above AND price < 1d EMA50 AND volume confirmation
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -20 from above
            if williams_r[i-1] >= -20 and williams_r[i] < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -80 from below
            if williams_r[i-1] <= -80 and williams_r[i] > -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals