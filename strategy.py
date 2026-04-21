#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian breakout with weekly trend filter and volume confirmation.
Longs when price breaks above 20-period high with weekly close above weekly SMA50 and volume > 1.5x average;
shorts when price breaks below 20-period low with weekly close below weekly SMA50 and volume > 1.5x average.
Exit on reversal signal (opposite breakout) or 2x ATR stop.
Designed for low-frequency, high-conviction trades to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        weekly_sma50 = sma50_1w_aligned[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above upper channel with weekly uptrend and volume
            if (price_high > upper_channel and 
                price_close > weekly_sma50 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower channel with weekly downtrend and volume
            elif (price_low < lower_channel and 
                  price_close < weekly_sma50 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite breakout OR ATR-based stoploss
            exit_signal = False
            
            # Opposite breakout exit
            if position == 1 and price_low < lower_channel:
                exit_signal = True
            elif position == -1 and price_high > upper_channel:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from channel level)
            if position == 1:
                # For longs, stop below lower channel minus 2x ATR
                if price_close < lower_channel - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above upper channel plus 2x ATR
                if price_close > upper_channel + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_DonchianBreakout_WeeklySMA50Trend_Volume1.5x_ATR2x"
timeframe = "6h"
leverage = 1.0