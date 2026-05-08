#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend(10,3) with 1d EMA(34) trend filter and 20-period volume spike.
# Trend follows only when 1d EMA(34) is rising and volume > 2x EMA(20) volume.
# Exits on Supertrend reversal or when 1d trend changes.
# Target: 40-80 total trades over 4 years (10-20/year) to minimize fee drag.

name = "4h_Supertrend_10_3_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d[1:] > ema_34_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 20-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Supertrend calculation on 4h data (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Final Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Align 1d indicators to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, atr_period)  # Ensure enough data for volume EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i]) or
            np.isnan(supertrend[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: uptrend on 1d, price above Supertrend, volume confirmation
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] > supertrend[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend on 1d, price below Supertrend, volume confirmation
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] < supertrend[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend reversal or 1d trend change to downtrend
            if (direction[i] == -1 or  # Supertrend turned down
                trend_up_aligned[i] <= 0.5):  # 1d trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend reversal or 1d trend change to uptrend
            if (direction[i] == 1 or  # Supertrend turned up
                trend_up_aligned[i] > 0.5):  # 1d trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals