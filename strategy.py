#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and weekly EMA21 trend filter
# - Long when price breaks above weekly Donchian high (10 periods) with volume expansion and price above weekly EMA21
# - Short when price breaks below weekly Donchian low (10 periods) with volume expansion and price below weekly EMA21
# - Exit when price crosses back below/above weekly EMA21
# - Volume filter requires current volume > 1.5x 10-period average
# - Weekly timeframe provides fewer, higher-quality signals to avoid overtrading
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_WeeklyDonchianBreakout_EMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and EMA calculations
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (10-period high/low)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Donchian high: rolling max of high over 10 periods
    donchian_high = pd.Series(high_weekly).rolling(window=10, min_periods=10).max().values
    # Donchian low: rolling min of low over 10 periods
    donchian_low = pd.Series(low_weekly).rolling(window=10, min_periods=10).min().values
    
    # Calculate weekly EMA21 for trend filter
    close_weekly = df_weekly['close'].values
    ema_21_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    ema_21_weekly_daily = align_htf_to_ltf(prices, df_weekly, ema_21_weekly)
    
    # Volume filters (daily timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (1.5 * vol_ma_10)  # Volume confirmation
    volume_expansion = volume > np.roll(volume, 1)  # Current volume > previous
    volume_expansion[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(ema_21_weekly_daily[i]) or np.isnan(volume_filter[i]) or np.isnan(volume_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume expansion and above EMA21
            if close[i] > donchian_high_daily[i] and volume_expansion[i] and close[i] > ema_21_weekly_daily[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volume expansion and below EMA21
            elif close[i] < donchian_low_daily[i] and volume_expansion[i] and close[i] < ema_21_weekly_daily[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA21
            if close[i] < ema_21_weekly_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA21
            if close[i] > ema_21_weekly_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals