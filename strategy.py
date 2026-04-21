#!/usr/bin/env python3
"""
6h_WeeklyDonchian_Breakout_1dTrend_Confirmation
Hypothesis: On 6h timeframe, buy when price breaks above weekly Donchian high (5-period) and daily close is above daily EMA50 (bullish trend filter). Sell when price breaks below weekly Donchian low and daily close is below daily EMA50. Use volume confirmation (volume > 1.5x 20-period average) to filter false breakouts. Target: 20-50 trades/year (80-200 over 4 years). Works in bull via breakouts, in bear via short breakdowns with trend filter preventing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Daily trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Load weekly data for Donchian channels (5-period = approx 1 month) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 5-period Donchian channels on weekly data
    # Highest high of last 5 weekly bars
    donchian_high = pd.Series(high_1w).rolling(window=5, min_periods=5).max().values
    # Lowest low of last 5 weekly bars
    donchian_low = pd.Series(low_1w).rolling(window=5, min_periods=5).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === Volume confirmation: 20-period volume average on 6h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        daily_trend = ema_50_1d_aligned[i]
        weekly_high = donchian_high_aligned[i]
        weekly_low = donchian_low_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + daily close above EMA50 + volume spike
            if (price_close > weekly_high and 
                price_close > daily_trend and 
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + daily close below EMA50 + volume spike
            elif (price_close < weekly_low and 
                  price_close < daily_trend and 
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to the opposite Donchian level (mean reversion within weekly range)
            if position == 1 and price_close < weekly_low:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > weekly_high:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dTrend_Confirmation"
timeframe = "6h"
leverage = 1.0