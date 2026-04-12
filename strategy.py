#!/usr/bin/env python3
"""
12h_1d_Multi_TF_Pullback_Reversal_v1
Hypothesis: Use daily EMA trend and 12h price pullbacks to EMA21 with volume confirmation.
Long when price pulls back to EMA21 in daily uptrend with volume spike, short when pulls back in daily downtrend.
Works in bull via pullbacks to support in uptrend, in bear via pullbacks to resistance in downtrend.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Multi_TF_Pullback_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Daily EMA21 for trend direction
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 12h EMA21 for pullback entries
    ema_21_12h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(ema_21_12h[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine daily trend
        daily_uptrend = close_1d[-1] > ema_21_1d[-1] if len(close_1d) > 0 else False
        daily_downtrend = close_1d[-1] < ema_21_1d[-1] if len(close_1d) > 0 else False
        
        # Pullback to EMA21 with volume
        pullback_to_ema = abs(close[i] - ema_21_12h[i]) / ema_21_12h[i] < 0.005  # within 0.5%
        volume_spike = vol_ratio[i] > 1.8
        
        # Entry conditions
        long_entry = daily_uptrend and pullback_to_ema and volume_spike and close[i] > ema_21_12h[i]
        short_entry = daily_downtrend and pullback_to_ema and volume_spike and close[i] < ema_21_12h[i]
        
        # Exit conditions: reverse signal or trend change
        exit_long = not daily_uptrend or (close[i] < ema_21_12h[i] and position == 1)
        exit_short = not daily_downtrend or (close[i] > ema_21_12h[i] and position == -1)
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals