#!/usr/bin/env python3
# Hypothesis: 12h Williams %R reversal with 1d EMA trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from below AND price > 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Short when Williams %R crosses below -20 from above AND price < 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Exit when Williams %R crosses the opposite threshold (-20 for long exit, -80 for short exit).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing mean reversals in ranging markets and avoiding strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R for mean reversion signals (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R parameters
    lookback = 14
    
    # Calculate highest high and lowest low over lookback period
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral value when range is zero
    )
    
    # Align Williams %R to 12h timeframe (already aligned to completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA50 for trend filter (HTF)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1d volume spike filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)  # Volume above 1.5x 20-period average
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND price > 1d EMA50 AND volume spike
            if (williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80 and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above AND price < 1d EMA50 AND volume spike
            elif (williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20 and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -20 (overbought threshold)
            if williams_r_aligned[i] < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -80 (oversold threshold)
            if williams_r_aligned[i] > -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals