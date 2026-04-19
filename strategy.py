#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_EMA200_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6-day (approx 48-period) EMA200 equivalent for 6h timeframe
    # Using 48-period EMA as proxy for longer-term trend on 6h chart
    ema48_6h = pd.Series(close).ewm(span=48, adjust=False, min_periods=48).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or \
           np.isnan(ema48_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bull_power = bull_power_6h[i]
        bear_power = bear_power_6h[i]
        ema48 = ema48_6h[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        bullish_momentum = bull_power > 0 and bull_power > abs(bear_power)
        bearish_momentum = bear_power < 0 and abs(bear_power) > bull_power
        above_trend = price > ema48
        below_trend = price < ema48
        
        if position == 0:
            # Long: Bullish momentum + above trend EMA + volume
            if bullish_momentum and above_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bearish momentum + below trend EMA + volume
            elif bearish_momentum and below_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish momentum appears or price crosses below EMA
            if bearish_momentum or price < ema48:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish momentum appears or price crosses above EMA
            if bullish_momentum or price > ema48:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals