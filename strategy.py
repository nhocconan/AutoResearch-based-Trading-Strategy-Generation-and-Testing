#!/usr/bin/env python3
"""
Hypothesis: 1-day timeframe strategy using weekly Donchian breakout with weekly ADX trend filter and volume confirmation.
Longs when price breaks above weekly Donchian upper channel with ADX>20 and volume>1.5x average;
shorts when price breaks below weekly Donchian lower channel with ADX>20 and volume>1.5x average.
Exit on price crossing back through weekly midline or 2x ATR stop.
Designed for 10-30 trades/year to minimize fee decay while capturing major trend moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 14-period ADX for trend filter
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
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume spike > 1.5x 20-period average (daily)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period daily)
    tr1_d = prices['high'].values - prices['low'].values
    tr2_d = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3_d = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2_d[0] = tr1_d[0]
    tr3_d[0] = tr1_d[0]
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_d = pd.Series(tr_d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr_d[i]
        
        if position == 0:
            # Enter long: break above upper channel with volume and trend
            if (price_high > upper and 
                adx_val > 20 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower channel with volume and trend
            elif (price_low < lower and 
                  adx_val > 20 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: midline cross OR ATR-based stoploss
            exit_signal = False
            
            # Midline exit
            if position == 1 and price_close < mid:
                exit_signal = True
            elif position == -1 and price_close > mid:
                exit_signal = True
            
            # ATR-based stoploss (2x ATR from channel level)
            if position == 1:
                # For longs, stop below lower channel
                if price_close < lower - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above upper channel
                if price_close > upper + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_1wDonchian_Breakout_1wADX20_Volume1.5x_ATR2x"
timeframe = "1d"
leverage = 1.0