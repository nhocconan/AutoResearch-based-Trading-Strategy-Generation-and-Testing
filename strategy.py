#!/usr/bin/env python3
"""
Hypothesis: 6-hour breakout above weekly Donchian(40) high with volume > 2x 60-period average and weekly EMA50 trend filter.
Weekly trend filter avoids counter-trend trades in bear markets.
Volume confirmation ensures breakout strength.
Short when price breaks below weekly Donchian low with volume > 2x average and weekly EMA50 < weekly SMA200.
Exit when price returns to the opposite weekly Donchian band.
Weekly timeframe provides stability; 6h entries capture timely breakouts.
Target: 60-120 total trades over 4 years (15-30/year) for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend and Donchian
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly SMA200
    sma_200 = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    
    # Calculate weekly Donchian channel (40-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=40, min_periods=40).max().values
    donchian_low = pd.Series(low_1w).rolling(window=40, min_periods=40).min().values
    
    # Align weekly indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    sma_200_aligned = align_htf_to_ltf(prices, df_1w, sma_200)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 60-period volume average on 6h timeframe
    vol_6h = prices['volume'].values
    vol_ma_60 = pd.Series(vol_6h).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(sma_200_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (6h close and volume)
        price_close = prices['close'].iloc[i]
        vol_current = prices['volume'].iloc[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high, volume > 2x avg, weekly EMA50 > SMA200 (uptrend)
            if (price_close > donchian_high_aligned[i] and 
                vol_current > 2.0 * vol_ma_60[i] and
                ema_50_aligned[i] > sma_200_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low, volume > 2x avg, weekly EMA50 < SMA200 (downtrend)
            elif (price_close < donchian_low_aligned[i] and 
                  vol_current > 2.0 * vol_ma_60[i] and
                  ema_50_aligned[i] < sma_200_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite weekly Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= weekly Donchian low
                if price_close <= donchian_low_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= weekly Donchian high
                if price_close >= donchian_high_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian40_Volume2x_EMA50_Trend"
timeframe = "6h"
leverage = 1.0