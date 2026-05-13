#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Use daily Camarilla pivot levels (R3/S3) as strong support/resistance. 
Breakout above R3 with 1d EMA34 uptrend and volume confirmation signals long. 
Breakdown below S3 with 1d EMA34 downtrend and volume confirmation signals short. 
Camarilla levels from higher timeframe provide robust S/R that works in both bull/bear markets, 
while volume confirmation filters false breakouts. Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (R3/S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    r3 = daily_pivot + (daily_range * 1.1 / 2)  # R3 = P + 1.1*(H-L)/2
    s3 = daily_pivot - (daily_range * 1.1 / 2)  # S3 = P - 1.1*(H-L)/2
    
    # Align daily R3/S3 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    uptrend_1d = daily_close > ema_34_1d
    downtrend_1d = daily_close < ema_34_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above R3, 1d uptrend, volume confirmation
            if close[i] > r3_aligned[i] and uptrend_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3, 1d downtrend, volume confirmation
            elif close[i] < s3_aligned[i] and downtrend_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below daily pivot or trend reverses
            pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
            if close[i] < pivot_aligned[i] or not uptrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above daily pivot or trend reverses
            pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
            if close[i] > pivot_aligned[i] or not downtrend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals