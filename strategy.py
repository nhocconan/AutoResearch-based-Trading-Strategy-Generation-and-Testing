#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ADX trend filter.
In uptrend (ADX > 25 and +DI > -DI), buy breakouts above 1d Donchian high (20-period).
In downtrend (ADX > 25 and -DI > +DI), sell breakouts below 1d Donchian low (20-period).
Volume must exceed 1.8x 30-period average to confirm breakout strength.
Exit on trend reversal (ADX < 20) or when price crosses the 1d Donchian midpoint.
Designed for 15-25 trades/year (60-100 total over 4 years) to minimize fee fade while capturing strong trending moves.
Works in bull markets via upward breakouts and in bear markets via downward breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Donchian and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high = high_roll
    donch_low = low_roll
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 1d ADX (14-period) for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for 1d bar to close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Volume confirmation (volume spike > 1.8x 30-period average)
    vol_ma_30 = pd.Series(prices['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio = prices['volume'].values / vol_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        high_price = donch_high_aligned[i]
        low_price = donch_low_aligned[i]
        mid_price = donch_mid_aligned[i]
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: breakout above Donchian high in uptrend (ADX>25 and +DI>-DI)
            if (price_close > high_price and 
                adx_val > 25 and 
                plus_di_val > minus_di_val and 
                vol_ratio_val > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown below Donchian low in downtrend (ADX>25 and -DI>+DI)
            elif (price_close < low_price and 
                  adx_val > 25 and 
                  minus_di_val > plus_di_val and 
                  vol_ratio_val > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (ADX < 20) or price crosses Donchian midpoint
            exit_signal = False
            
            # Trend reversal exit
            if adx_val < 20:
                exit_signal = True
            
            # Midpoint cross exit
            if position == 1 and price_close < mid_price:
                exit_signal = True
            elif position == -1 and price_close > mid_price:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_ADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0