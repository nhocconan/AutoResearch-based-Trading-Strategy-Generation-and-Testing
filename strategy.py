#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily volume confirmation and EMA trend filter
# Long when price breaks above Donchian(20) high with above-average daily volume and price above daily EMA50
# Short when price breaks below Donchian(20) low with above-average daily volume and price below daily EMA50
# Uses discrete position sizing (0.25) to minimize fee churn
# Designed to capture strong trends while avoiding choppy markets via volume and trend filters
# Target: 20-50 trades per year to stay within optimal range for 4h timeframe

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and trend filters
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        for i in range(19, n):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate current daily volume for volume filter
    vol_daily_current = np.full(len(df_daily), np.nan)
    for i in range(len(df_daily)):
        vol_daily_current[i] = df_daily.iloc[i]['volume']
    
    vol_daily_current_aligned = align_htf_to_ltf(prices, df_daily, vol_daily_current)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(vol_daily_current_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-period average
        vol_filter = vol_daily_current_aligned[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        # Determine trend direction
        bullish_trend = close[i] > ema50_daily_aligned[i]
        bearish_trend = close[i] < ema50_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with volume and trend filter
            long_condition = (
                close[i] > donchian_high[i] and   # price breaks above Donchian high
                bullish_trend and                 # only long in uptrend
                vol_filter                        # volume confirmation
            )
            
            short_condition = (
                close[i] < donchian_low[i] and    # price breaks below Donchian low
                bearish_trend and                 # only short in downtrend
                vol_filter                        # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or trend changes
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < donchian_mid or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or trend changes
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > donchian_mid or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals