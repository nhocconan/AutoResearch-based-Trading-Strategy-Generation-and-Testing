#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h volume spike and EMA trend filter.
Longs when price breaks above upper Donchian with volume>1.5x average and price>12h EMA50.
Shorts when price breaks below lower Donchian with volume>1.5x average and price<12h EMA50.
Exit on price crossing back through midpoint or 2x ATR stop.
Designed for 20-40 trades/year to minimize fee drag while capturing breakouts with trend and volume confirmation.
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
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
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
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = high_20[i]
        lower = low_20[i]
        midpoint = donchian_mid[i]
        ema_val = ema_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume and trend
            if (price_high > upper and 
                vol_ratio_val > 1.5 and 
                price_close > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume and trend
            elif (price_low < lower and 
                  vol_ratio_val > 1.5 and 
                  price_close < ema_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: midpoint cross OR ATR-based stoploss
            exit_signal = False
            
            # Midpoint exit
            if position == 1 and price_close < midpoint:
                exit_signal = True
            elif position == -1 and price_close > midpoint:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from breakout level)
            if position == 1:
                # For longs, stop below lower band minus 2x ATR
                if price_close < lower - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above upper band plus 2x ATR
                if price_close > upper + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume1.5x_12hEMA50_Trend_MidpointExit"
timeframe = "4h"
leverage = 1.0