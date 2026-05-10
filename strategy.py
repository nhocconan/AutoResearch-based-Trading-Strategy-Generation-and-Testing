#!/usr/bin/env python3
# 4h_Chaikin_Money_Flow_Breakout_1dTrend_Volume
# Hypothesis: Uses Chaikin Money Flow (20) on 1d to detect institutional accumulation/distribution.
# Enters long when price breaks above previous day's high with CMF > 0.25 and 1d uptrend (close > EMA34).
# Enters short when price breaks below previous day's low with CMF < -0.25 and 1d downtrend (close < EMA34).
# Exits when price returns to the previous day's close or trend reverses.
# Uses 1-day EMA34 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_Chaikin_Money_Flow_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for CMF and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Chaikin Money Flow (20) on 1d
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Avoid division by zero
    price_range = high_1d - low_1d
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / price_range
    mfv = mfm * volume_1d
    
    # 20-period sums
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    
    # Avoid division by zero
    volume_sum = np.where(volume_sum == 0, 1e-10, volume_sum)
    cmf_20 = mfv_sum / volume_sum
    
    # Align CMF and EMA to 4h
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20)
    
    # Use previous day's high/low/close for breakout levels
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Align breakout levels to 4h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(cmf_20_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above previous day's high with CMF > 0.25 and uptrend
            if (close[i] > prev_high_aligned[i] and 
                cmf_20_aligned[i] > 0.25 and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below previous day's low with CMF < -0.25 and downtrend
            elif (close[i] < prev_low_aligned[i] and 
                  cmf_20_aligned[i] < -0.25 and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to previous day's close or trend reverses
            if (close[i] <= prev_close_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to previous day's close or trend reverses
            if (close[i] >= prev_close_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals