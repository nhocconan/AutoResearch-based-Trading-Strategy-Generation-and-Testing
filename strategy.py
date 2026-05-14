#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and 1d volume spike confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (EMA50), reversions from extremes capture swing points.
# Volume spike confirms institutional participation. Works in bull/bear by adapting to trend direction via EMA50.
# Long: %R crosses above -80 from below AND close > EMA50_1d AND volume > 2.0 * 20-period average volume.
# Short: %R crosses below -20 from above AND close < EMA50_1d AND volume > 2.0 * 20-period average volume.
# Exit: %R crosses above -20 (for longs) or below -80 (for shorts) to lock in profits before reversal.
# Discrete sizing 0.25 to balance capture and drawdown. Target: 80-120 total trades over 4 years (20-30/year).

name = "6h_WilliamsR_Reversal_1dEMA50_Trend_1dVolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Williams %R (14-period) on 6h timeframe
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
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: %R crosses above -80 from below AND close > EMA50_1d AND volume confirmation
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and
                close[i] > ema_50_1d_aligned[i] and
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: %R crosses below -20 from above AND close < EMA50_1d AND volume confirmation
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and
                  close[i] < ema_50_1d_aligned[i] and
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: %R crosses above -20 (taking profit before overbought reversal)
            if williams_r[i-1] < -20 and williams_r[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: %R crosses below -80 (taking profit before oversold reversal)
            if williams_r[i-1] > -80 and williams_r[i] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals