#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_BullBear_Power_With_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Smooth the power values with 3-period EMA
    bull_power_smooth = pd.Series(bull_power_1d).ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power_1d).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    # 6h trend filter: EMA(34) slope
    close_s = pd.Series(close)
    ema_34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34 - np.roll(ema_34, 1)
    ema_34_slope[0] = 0
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(ema_34_slope[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: bullish when EMA slope > 0
        bullish_trend = ema_34_slope[i] > 0
        bearish_trend = ema_34_slope[i] < 0
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) with volume and bullish trend
            if bull_power_aligned[i] > 0 and volume_ok and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) with volume and bearish trend
            elif bear_power_aligned[i] < 0 and volume_ok and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power <= 0 or trend turns bearish
            if bull_power_aligned[i] <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power >= 0 or trend turns bullish
            if bear_power_aligned[i] >= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals