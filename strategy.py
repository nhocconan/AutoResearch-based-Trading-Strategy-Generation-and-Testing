#!/usr/bin/env python3
name = "4h_WaterLevel_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily price range for water level calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Water Level = (Daily High + Daily Low) / 2
    water_level = (daily_high + daily_low) / 2
    water_level_aligned = align_htf_to_ltf(prices, df_1d, water_level)
    
    # 1-period change in water level to detect expansion/contraction
    water_level_change = np.abs(np.diff(water_level, prepend=water_level[0]))
    water_level_change_aligned = align_htf_to_ltf(prices, df_1d, water_level_change)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 6)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(water_level_aligned[i]) or 
            np.isnan(water_level_change_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > vol_ma_6[i] * 1.5
        
        if position == 0:
            # Long: water level expanding AND price above water level in daily uptrend
            expanding = water_level_change_aligned[i] > water_level_change_aligned[i-1]
            if expanding and close[i] > water_level_aligned[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: water level expanding AND price below water level in daily downtrend
            elif expanding and close[i] < water_level_aligned[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below water level or volume drops
            if close[i] < water_level_aligned[i] or volume[i] < vol_ma_6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above water level or volume drops
            if close[i] > water_level_aligned[i] or volume[i] < vol_ma_6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Water level (midpoint of daily range) acts as dynamic support/resistance
# - Water level expansion indicates increasing volatility and potential breakout
# - Break above/below water level with volume in direction of daily trend = entry
# - Works in bull markets (buy water level breaks in uptrend) and bear markets (sell water level breaks in downtrend)
# - Volume confirmation (1.5x average) filters false breakouts
# - Exit when price returns to water level or volume diminishes
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Water level provides cleaner signal than pivot points in ranging markets