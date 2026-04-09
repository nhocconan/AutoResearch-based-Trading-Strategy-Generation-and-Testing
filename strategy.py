#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_v2
# Hypothesis: Weekly Donchian breakout with daily trend filter and volume confirmation.
# Long when price breaks above weekly Donchian high and daily close > daily EMA200.
# Short when price breaks below weekly Donchian low and daily close < daily EMA200.
# Uses volume confirmation (volume > 1.5x 20-day average) to filter breakouts.
# Designed for low-frequency, high-conviction trades to avoid fee drag in bear markets.
# Target: 10-25 trades/year (40-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. Weekly Donchian channels (20-period)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate Donchian channels on weekly data
    donchian_high = np.full(len(weekly_high), np.nan)
    donchian_low = np.full(len(weekly_low), np.nan)
    
    for i in range(20, len(weekly_high)):
        donchian_high[i] = np.max(weekly_high[i-20:i])
        donchian_low[i] = np.min(weekly_low[i-20:i])
    
    # Align to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # 2. Daily EMA200 for trend filter
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 3. Volume confirmation (20-day average)
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(ema200[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low
            if close[i] < donchian_low_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high
            if close[i] > donchian_high_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with trend and volume
            if (close[i] > donchian_high_daily[i] and 
                close[i] > ema200[i] and vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with trend and volume
            elif (close[i] < donchian_low_daily[i] and 
                  close[i] < ema200[i] and vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals