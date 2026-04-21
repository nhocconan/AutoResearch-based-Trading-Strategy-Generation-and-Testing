#!/usr/bin/env python3
"""
Hypothesis: 1d strategy combining 1-week Donchian channel breakout with 1-week EMA trend filter and volume confirmation.
Breakouts above weekly Donchian upper (bullish) or below lower (bearish) signal momentum shifts.
Only trade long when price > weekly EMA50 (uptrend), short when price < weekly EMA50 (downtrend).
Volume confirmation requires >1.5x 20-period average volume to filter weak breakouts.
Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trends.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for Donchian, EMA, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1-week volume confirmation: volume > 1.5x 20-period average
    vol_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1w / vol_ma_20
    
    # Align all 1w indicators to 1d timeframe (wait for weekly bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + uptrend (price > EMA50) + volume spike
            if (price_close > donch_high and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + downtrend (price < EMA50) + volume spike
            elif (price_close < donch_low and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA50 in opposite direction) or loss of momentum
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0