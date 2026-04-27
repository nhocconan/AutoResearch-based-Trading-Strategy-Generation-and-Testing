#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1-week EMA trend filter and volume confirmation.
# Long when price breaks above 20-period weekly Donchian high with weekly EMA20 uptrend and volume > 1.5x average.
# Short when price breaks below 20-period weekly Donchian low with weekly EMA20 downtrend and volume > 1.5x average.
# Exit when price crosses the weekly Donchian midline (average of high and low over 20 periods).
# Targets low trade frequency (7-25/year) to avoid fee drag, suitable for 1d timeframe.
# Works in bull markets via trend-following breakouts and in bear via short-side breakdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate weekly EMA20 for trend filter
    ema_period = 20
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period - 1] = np.mean(close_weekly[:ema_period])
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * (2 / (ema_period + 1)) + 
                             ema_weekly[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate weekly Donchian channels (20-period)
    donch_period = 20
    donch_high = np.full(len(close_weekly), np.nan)
    donch_low = np.full(len(close_weekly), np.nan)
    donch_mid = np.full(len(close_weekly), np.nan)
    
    for i in range(donch_period - 1, len(close_weekly)):
        donch_high[i] = np.max(high_weekly[i - donch_period + 1:i + 1])
        donch_low[i] = np.min(low_weekly[i - donch_period + 1:i + 1])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Align weekly indicators to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    donch_high_aligned = align_htf_to_ltf(prices, df_weekly, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_weekly, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_weekly, donch_mid)
    
    # Volume MA for confirmation (20-period weekly)
    vol_ma_weekly = np.full(len(volume_weekly), np.nan)
    for i in range(19, len(volume_weekly)):
        vol_ma_weekly[i] = np.mean(volume_weekly[i - 19:i + 1])
    vol_ma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly Donchian channels, EMA20, and volume MA20
    start_idx = max(donch_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_weekly_aligned[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above weekly Donchian high with weekly EMA20 uptrend and volume filter
            if (price > donch_high_aligned[i] and 
                price > ema_weekly_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below weekly Donchian low with weekly EMA20 downtrend and volume filter
            elif (price < donch_low_aligned[i] and 
                  price < ema_weekly_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Donchian midline
            if price < donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly Donchian midline
            if price > donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyEMA20_Volume"
timeframe = "1d"
leverage = 1.0