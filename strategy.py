#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1w EMA50 trend filter and 1d volume spike confirmation.
# Williams %R identifies overbought/oversold conditions. Long when %R crosses above -80 from below (oversold bounce) AND 1w EMA50 uptrend AND 1d volume > 1.5 * 20-period average volume.
# Short when %R crosses below -20 from above (overbought rejection) AND 1w EMA50 downtrend AND 1d volume > 1.5 * 20-period average volume.
# Exit when %R crosses -50 (mean reversion midpoint) or opposite signal triggers.
# Uses discrete position sizing (0.25) to limit fee churn. Target: 80-180 total trades over 4 years (20-45/year) for 6h.

name = "6h_WilliamsR_Reversal_1wEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_trend_1w = np.where(ema_50_1w > np.roll(ema_50_1w, 1), 1.0, -1.0)  # 1 for uptrend, -1 for downtrend
    ema50_trend_1w[0] = 1.0  # default to uptrend for first value
    ema50_trend_aligned = align_htf_to_ltf(prices, df_1w, ema50_trend_1w)
    
    # Calculate 1d volume confirmation filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # start after Williams %R warmup
        # Skip if any required data is NaN
        if (np.isnan(ema50_trend_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(williams_r[i-1])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below (oversold bounce) AND 1w EMA50 uptrend AND volume confirmation
            if (williams_r[i-1] <= -80 and williams_r[i] > -80 and 
                ema50_trend_aligned[i] > 0 and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above (overbought rejection) AND 1w EMA50 downtrend AND volume confirmation
            elif (williams_r[i-1] >= -20 and williams_r[i] < -20 and 
                  ema50_trend_aligned[i] < 0 and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion) or short signal triggers
            if williams_r[i] > -50 or (williams_r[i-1] >= -20 and williams_r[i] < -20 and ema50_trend_aligned[i] < 0 and volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion) or long signal triggers
            if williams_r[i] < -50 or (williams_r[i-1] <= -80 and williams_r[i] > -80 and ema50_trend_aligned[i] > 0 and volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals