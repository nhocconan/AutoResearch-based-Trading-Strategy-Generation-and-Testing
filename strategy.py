#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1w Donchian breakout with 1w ADX trend filter and volume confirmation.
Breakouts above 1w Donchian upper channel (20-period) trigger longs when 1w ADX > 25 (trending).
Breakouts below 1w Donchian lower channel trigger shorts when 1w ADX > 25.
Volume must exceed 1.5x 20-period average to confirm breakout strength.
Exit on Donchian middle band cross or 2x ATR stop.
Designed for 15-30 trades/year (60-120 total over 4 years) to minimize fee fade while capturing strong trends.
Works in bull markets via upward breakouts and in bear markets via downward breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 14-period ADX on weekly data
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    plus_dm[1:] = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                           np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm[1:] = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                            np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w Donchian and 1w ADX to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        mid = donch_mid_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume and trend
            if (price_high > upper and 
                adx_val > 25 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume and trend
            elif (price_low < lower and 
                  adx_val > 25 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: middle band cross OR ATR-based stoploss
            exit_signal = False
            
            # Middle band exit
            if position == 1 and price_close < mid:
                exit_signal = True
            elif position == -1 and price_close > mid:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from extreme)
            if position == 1:
                if price_close < upper - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if price_close > lower + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_1wDonchian_Breakout_1wADX25_Volume1.5x_ATR2"
timeframe = "12h"
leverage = 1.0