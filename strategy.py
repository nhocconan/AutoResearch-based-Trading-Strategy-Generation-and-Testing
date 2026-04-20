# This strategy implements a 12-hour breakout system using weekly Donchian channels and weekly trend filters.
# It captures major trend moves while avoiding whipsaws through multi-timeframe confirmation.
# The strategy works in both bull and bear markets by focusing on breakouts with trend alignment.
# Position sizing is conservative (0.25) to manage drawdowns during choppy periods.
# Entry conditions: Price breaks above weekly Donchian high AND weekly EMA34 > weekly EMA89 (long)
#                 OR Price breaks below weekly Donchian low AND weekly EMA34 < weekly EMA89 (short)
# Exit conditions: Price crosses back to weekly EMA34 or opposite Donchian breakout occurs
# Volume confirmation is used to filter breakouts (volume > 1.5x 20-period average)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data once before loop (HTF = 1w)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 90:  # Need enough data for EMA89
        return np.zeros(n)
    
    # Calculate weekly indicators
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Weekly Donchian channels (20-period)
    donchian_high = np.full(len(weekly_high), np.nan)
    donchian_low = np.full(len(weekly_low), np.nan)
    for i in range(20, len(weekly_high)):
        donchian_high[i] = np.max(weekly_high[i-20:i])
        donchian_low[i] = np.min(weekly_low[i-20:i])
    
    # Weekly EMAs for trend filter
    def calculate_ema(data, period):
        ema = np.full_like(data, np.nan)
        if len(data) < period:
            return ema
        multiplier = 2 / (period + 1)
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema34 = calculate_ema(weekly_close, 34)
    ema89 = calculate_ema(weekly_close, 89)
    
    # Align weekly indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34)
    ema89_aligned = align_htf_to_ltf(prices, df_weekly, ema89)
    
    # Volume confirmation on 12h timeframe
    volume = prices['volume'].values
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where we have all data
    start_idx = max(100, 20)  # Ensure we have enough warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(ema89_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + uptrend + volume confirmation
            if (price > donchian_high_aligned[i] and 
                ema34_aligned[i] > ema89_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low + downtrend + volume confirmation
            elif (price < donchian_low_aligned[i] and 
                  ema34_aligned[i] < ema89_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA34 or breaks below opposite Donchian low
            if (price < ema34_aligned[i] or 
                price < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA34 or breaks above opposite Donchian high
            if (price > ema34_aligned[i] or 
                price > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchian_EMA_Trend"
timeframe = "12h"
leverage = 1.0