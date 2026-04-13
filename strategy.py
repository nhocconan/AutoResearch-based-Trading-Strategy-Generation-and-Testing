#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (EMA13 on 6h)
# Trend filter: 1d EMA50 - only go long when price > EMA50, short when price < EMA50
# Volume confirmation: current volume > 1.3x average volume (20-period)
# Elder Ray identifies institutional buying/selling pressure
# EMA50 trend filter ensures we trade with higher timeframe trend
# Volume confirmation reduces false signals
# Works in bull markets (strong Bull Power) and bear markets (strong Bear Power)
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA13 for Elder Ray
    ema13 = np.full(n, np.nan)
    if len(close) >= 13:
        close_series = pd.Series(close)
        ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    # 6h Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        close_1d_series = pd.Series(close_1d)
        ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Average volume (20-period = 20*6h = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    if len(volume) >= 20:
        volume_series = pd.Series(volume)
        avg_volume = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema50 = ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + above EMA50 + volume confirmation
            if (bull_power[i] > 0 and 
                price > ema50 and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power > 0 (selling pressure) + below EMA50 + volume confirmation
            elif (bear_power[i] > 0 and 
                  price < ema50 and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bear Power > 0 (selling pressure takes over) or below EMA50
            if (bear_power[i] > 0 or
                price < ema50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bull Power > 0 (buying pressure takes over) or above EMA50
            if (bull_power[i] > 0 or
                price > ema50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0