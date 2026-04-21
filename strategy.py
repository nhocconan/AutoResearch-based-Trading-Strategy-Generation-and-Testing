#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Uses 1d EMA50 to filter trend direction and avoid counter-trend trades. Designed for low trade frequency
(12-37 trades/year) to minimize fee drag while capturing strong momentum moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA50 trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_1d_aligned[i]
        upper_band = donch_high_aligned[i]
        lower_band = donch_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, uptrend
            if (price_close > upper_band and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, downtrend
            elif (price_close < lower_band and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Donchian band or trend reversal
            if position == 1 and (price_close < lower_band or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > upper_band or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_DonchianBreakout_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0