#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation
# Uses weekly Donchian(20) high/low to establish long-term trend direction
# Enters on breakout of daily Donchian(20) in direction of weekly trend
# Target: 15-25 trades/year per symbol, works in bull/bear via trend filter
# Avoids counter-trend trades and reduces whipsaw in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily Donchian(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian high/low (20-period)
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly Donchian(20) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    weekly_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: price above weekly Donchian high = uptrend, below low = downtrend
    weekly_close = df_1w['close'].values
    weekly_uptrend = weekly_close > weekly_high
    weekly_downtrend = weekly_close < weekly_low
    
    # Volume spike filter (20-period on daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to daily timeframe
    daily_high_alg = align_htf_to_ltf(prices, df_1d, high_roll)
    daily_low_alg = align_htf_to_ltf(prices, df_1d, low_roll)
    weekly_uptrend_alg = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_alg = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(daily_high_alg[i]) or np.isnan(daily_low_alg[i]) or
            np.isnan(weekly_uptrend_alg[i]) or np.isnan(weekly_downtrend_alg[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily breakout above Donchian high + weekly uptrend + volume spike
            if (close[i] > daily_high_alg[i] and weekly_uptrend_alg[i] > 0.5 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Daily breakout below Donchian low + weekly downtrend + volume spike
            elif (close[i] < daily_low_alg[i] and weekly_downtrend_alg[i] > 0.5 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < daily_low_alg[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > daily_high_alg[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0