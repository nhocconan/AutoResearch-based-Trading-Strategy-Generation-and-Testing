#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA34 trend filter + volume spike
# Williams %R(14) < -80 for oversold (long), > -20 for overbought (short)
# 1d EMA34 provides trend filter: long only when price > EMA34, short only when price < EMA34
# Volume confirmation: current volume > 2.0x 20-period MA to ensure conviction
# Uses mean reversion in extreme conditions with trend filter to avoid fighting the trend
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Williams %R(14) on 12h
    if len(close) >= 14:
        # Highest high and lowest low over 14 periods
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        
        # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        williams_r = np.where(
            (highest_high - lowest_low) != 0,
            ((highest_high - close) / (highest_high - lowest_low)) * -100,
            np.nan
        )
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend filter) AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend filter) AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold) OR price < EMA34 (trend change)
            if williams_r[i] > -50 or close[i] <= ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought) OR price > EMA34 (trend change)
            if williams_r[i] < -50 or close[i] >= ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals