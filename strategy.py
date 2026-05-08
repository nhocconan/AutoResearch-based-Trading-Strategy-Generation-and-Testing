#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Uses 1d EMA50 for trend direction, Donchian(20) breakout for entry, and volume > 1.5x 20-period average for confirmation.
# Designed to capture strong trends in both bull and bear markets while avoiding false breakouts in low-volume conditions.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Donchian_1dEMA50_VolumeConfirm"
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
    
    # Get daily data for EMA trend filter
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
    
    # Calculate daily ATR(14) for Donchian width context
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    atr14_daily = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr14_daily[13] = np.nanmean(tr[:14])
        for i in range(14, len(tr)):
            if np.isnan(atr14_daily[i-1]):
                atr14_daily[i] = np.nanmean(tr[i-13:i+1])
            else:
                atr14_daily[i] = (atr14_daily[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily Donchian channels (20-period)
    highest_high_20 = np.full(len(close_daily), np.nan)
    lowest_low_20 = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 20:
        for i in range(20, len(close_daily)):
            highest_high_20[i] = np.max(high_daily[i-19:i+1])
            lowest_low_20[i] = np.min(low_daily[i-19:i+1])
    
    # Calculate daily volume average for confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_daily, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_daily, lowest_low_20)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
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
        if (np.isnan(ema50_daily_aligned[i]) or np.isnan(highest_high_20_aligned[i]) or
            np.isnan(lowest_low_20_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average of daily volume
        vol_confirm = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_4h_current = volume[i]
            vol_confirm = vol_4h_current > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout in direction of daily EMA trend with volume confirmation
            # Long when price breaks above 20-day high and above daily EMA50 (bullish bias)
            long_condition = (
                high[i] > highest_high_20_aligned[i] and   # price breaks above 20-day high
                close[i] > ema50_daily_aligned[i] and      # price above daily EMA50 (bullish bias)
                vol_confirm                                # volume confirmation
            )
            
            # Short when price breaks below 20-day low and below daily EMA50 (bearish bias)
            short_condition = (
                low[i] < lowest_low_20_aligned[i] and      # price breaks below 20-day low
                close[i] < ema50_daily_aligned[i] and      # price below daily EMA50 (bearish bias)
                vol_confirm                                # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below 20-day low or below daily EMA50
            if low[i] < lowest_low_20_aligned[i] or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 20-day high or above daily EMA50
            if high[i] > highest_high_20_aligned[i] or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals