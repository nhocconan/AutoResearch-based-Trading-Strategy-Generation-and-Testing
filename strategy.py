#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullPower_Trend_v1"
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
    
    # Get daily data for Elder Ray calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for daily data
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power (Elder Ray)
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align daily Elder Ray components to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 6-period EMA for short-term trend on 6h data
    close_series = pd.Series(close)
    ema6 = close_series.ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(ema6[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        ema13 = ema13_1d_aligned[i]
        ema6_val = ema6[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) + price above short-term EMA + volume
            if bull_power > 0 and price > ema6_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + price below short-term EMA + volume
            elif bear_power < 0 and price < ema6_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull Power turns negative OR price drops below short-term EMA
            if bull_power <= 0 or price < ema6_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear Power turns positive OR price rises above short-term EMA
            if bear_power >= 0 or price > ema6_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals