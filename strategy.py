#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with weekly trend filter and volume confirmation.
Longs when price breaks above 20-day high with weekly EMA(21) upward and volume > 1.5x average;
shorts when price breaks below 20-day low with weekly EMA(21) downward and volume > 1.5x average.
Exit on price crossing back through 20-day midpoint or 2x ATR stop.
Designed for 7-25 trades/year to minimize fee dust while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    weekly_ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema_21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_21)
    
    # Calculate 20-day Donchian channels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period high and low
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 1d timeframe (prices are daily)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # ATR for stoploss (20-period on daily data)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation - 20-day average volume
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_ema_21_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        high_don = donchian_high_aligned[i]
        low_don = donchian_low_aligned[i]
        mid_don = donchian_mid_aligned[i]
        weekly_ema_val = weekly_ema_21_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        atr_val = atr_aligned[i]
        
        # Current volume
        volume_current = volume_1d[i]
        
        if position == 0:
            # Enter long: break above 20-day high with upward weekly trend and volume spike
            if (price_high > high_don and 
                weekly_ema_val > weekly_ema_21_aligned[i-1] and  # Weekly EMA rising
                volume_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Enter short: break below 20-day low with downward weekly trend and volume spike
            elif (price_low < low_don and 
                  weekly_ema_val < weekly_ema_21_aligned[i-1] and  # Weekly EMA falling
                  volume_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: midpoint cross OR ATR-based stoploss
            exit_signal = False
            
            # Midpoint exit
            if position == 1 and price_close < mid_don:
                exit_signal = True
            elif position == -1 and price_close > mid_don:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from breakout level)
            if position == 1:
                # For longs, stop below entry area
                if price_close < high_don - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above entry area
                if price_close > low_don + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA21_Trend_Volume1.5x_ATR2x"
timeframe = "1d"
leverage = 1.0