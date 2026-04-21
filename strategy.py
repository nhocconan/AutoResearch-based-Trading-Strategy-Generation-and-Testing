#!/usr/bin/env python3
"""
Hypothesis: 1d price closes above/below the weekly Donchian channel (20-period) with volume spike and ADX trend filter.
Long when close > weekly upper band, ADX>20, volume>1.5x average; short when close < weekly lower band, ADX>20, volume>1.5x average.
Exit when price crosses back through the weekly midpoint or 2x ATR stop.
Designed for 10-25 trades/year to minimize fee decay while capturing strong weekly trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for Donchian levels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 20-period weekly Donchian channel
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    midpoint_1w = (upper_1w + lower_1w) / 2
    
    # Calculate 14-period weekly ADX for trend filter
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
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    midpoint_1w_aligned = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume spike > 1.5x 20-period average (daily)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    # ATR for stoploss (20-period daily)
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
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(midpoint_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        upper = upper_1w_aligned[i]
        lower = lower_1w_aligned[i]
        midpoint = midpoint_1w_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: close above weekly upper band with volume and trend
            if (price_close > upper and 
                adx_val > 20 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: close below weekly lower band with volume and trend
            elif (price_close < lower and 
                  adx_val > 20 and 
                  vol_ratio_val > 1.5):
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
            
            # ATR-based stoploss (2x ATR from entry zone)
            if position == 1:
                # For longs, stop below lower band (as proxy for entry area)
                if price_close < lower - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above upper band (as proxy for entry area)
                if price_close > upper + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian20_ADX20_Volume1.5x_ATR2x"
timeframe = "1d"
leverage = 1.0