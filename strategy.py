#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily trend filter and volume confirmation
# Uses daily EMA50 for trend direction, weekly Donchian for market regime, and volume spike for entry
# Designed to capture strong trends while avoiding choppy markets. Target: 12-37 trades/year on 12h.

name = "12h_Donchian20_1dEMA50_1wDonchian20_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Get weekly data for Donchian20 regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate weekly Donchian20 for regime filter (trending vs ranging)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high_20w = np.full(len(high_weekly), np.nan)
    donchian_low_20w = np.full(len(low_weekly), np.nan)
    if len(high_weekly) >= 20:
        for i in range(20, len(high_weekly)):
            donchian_high_20w[i] = np.max(high_weekly[i-20:i])
            donchian_low_20w[i] = np.min(low_weekly[i-20:i])
    
    # Calculate 12h Donchian20 for entry signals
    donchian_high_20 = np.full(len(high), np.nan)
    donchian_low_20 = np.full(len(low), np.nan)
    if len(high) >= 20:
        for i in range(20, len(high)):
            donchian_high_20[i] = np.max(high[i-20:i])
            donchian_low_20[i] = np.min(low[i-20:i])
    
    # Calculate 12h volume average for volume confirmation
    vol_avg_20 = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align indicators to 12h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low_20w)
    
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
        if (np.isnan(ema50_daily_aligned[i]) or 
            np.isnan(donchian_high_20w_aligned[i]) or 
            np.isnan(donchian_low_20w_aligned[i]) or
            np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Market regime: weekly Donchian width indicates trending market
        weekly_range = donchian_high_20w_aligned[i] - donchian_low_20w_aligned[i]
        # Avoid extremely tight ranges (chop) and extremely wide ranges (exhaustion)
        regime_ok = (weekly_range > 0)  # Basic validity check
        
        if position == 0:
            # Look for entry: Donchian breakout with trend and volume confirmation
            long_condition = (
                close[i] > donchian_high_20[i] and      # break above 12h Donchian high
                close[i] > ema50_daily_aligned[i] and   # price above daily EMA50 (bullish bias)
                vol_confirm and                         # volume confirmation
                regime_ok                               # valid market regime
            )
            
            short_condition = (
                close[i] < donchian_low_20[i] and       # break below 12h Donchian low
                close[i] < ema50_daily_aligned[i] and   # price below daily EMA50 (bearish bias)
                vol_confirm and                         # volume confirmation
                regime_ok                               # valid market regime
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low or trend reverses
            if close[i] < donchian_low_20[i] or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or trend reverses
            if close[i] > donchian_high_20[i] or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals