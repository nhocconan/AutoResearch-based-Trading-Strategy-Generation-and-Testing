#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Longs when price breaks above 20-period high with volume > 1.5x average and 12h EMA(50) rising.
Shorts when price breaks below 20-period low with volume > 1.5x average and 12h EMA(50) falling.
Exit on opposite Donchian break or 2x ATR stop.
Designed for 20-40 trades/year to minimize fee drag while capturing high-probability momentum.
Works in both bull (breakouts) and bear (breakdowns) markets via symmetric logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 50-period EMA on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA slope: current > previous indicates rising trend
    ema_slope = np.zeros_like(ema_12h)
    ema_slope[1:] = ema_12h[1:] > ema_12h[:-1]
    
    # Align EMA slope to 4h timeframe
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    
    # Donchian channels (20-period high/low)
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume spike > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_slope_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_slope_val = ema_slope_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above 20-period high with volume and rising 12h EMA
            if (price_high > high_20[i] and 
                vol_ratio_val > 1.5 and 
                ema_slope_val):
                signals[i] = 0.25
                position = 1
            # Enter short: break below 20-period low with volume and falling 12h EMA
            elif (price_low < low_20[i] and 
                  vol_ratio_val > 1.5 and 
                  not ema_slope_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite Donchian break OR ATR-based stoploss
            exit_signal = False
            
            # Opposite Donchian break
            if position == 1 and price_low < low_20[i]:
                exit_signal = True
            elif position == -1 and price_high > high_20[i]:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from entry level)
            if position == 1:
                # For longs, stop below entry area (use 20-period low as reference)
                if price_close < low_20[i] - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above entry area (use 20-period high as reference)
                if price_close > high_20[i] + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume1.5x_12hEMA50Slope"
timeframe = "4h"
leverage = 1.0