#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and weekly trend filter
# Long when price breaks above 24-period (12-day) Donchian high with volume spike and weekly bullish trend
# Short when price breaks below 24-period Donchian low with volume spike and weekly bearish trend
# Exit on opposite Donchian band touch
# Uses weekly trend to avoid counter-trend trades in bear markets
# Target: 15-30 trades per symbol over 4 years (4-7.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and weekly data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate 12h Donchian channels (24-period lookback for 12-day equivalent)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=24, min_periods=24).max().values
    donchian_low = pd.Series(low_12h).rolling(window=24, min_periods=24).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h volume average (24-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for 24-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume[i]  # Current 12h volume
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike and weekly bullish trend
            if (price > donchian_high_aligned[i] and 
                vol_12h_current > 1.8 * vol_ma_12h_aligned[i] and  # Volume spike
                price > ema_weekly_aligned[i]):                    # Price above weekly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with volume spike and weekly bearish trend
            elif (price < donchian_low_aligned[i] and 
                  vol_12h_current > 1.8 * vol_ma_12h_aligned[i] and  # Volume spike
                  price < ema_weekly_aligned[i]):                    # Price below weekly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches Donchian low (opposite band)
            if price <= donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches Donchian high (opposite band)
            if price >= donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0