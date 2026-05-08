#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d RSI filter and volume confirmation
# Uses Donchian(20) breakouts for directional entries, filtered by 1d RSI(14) to avoid overextended moves,
# and confirmed by volume spikes (>1.5x 20-period average). Designed to capture trends in both bull and bear markets
# while avoiding choppy, low-volume environments. Target: 25-40 trades/year.

name = "4h_Donchian20_1dRSI14_VolumeConfirm"
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
    
    # Get daily data for RSI filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_daily = df_daily['close'].values
    delta = np.diff(close_daily)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_daily), np.nan)
    avg_loss = np.full(len(close_daily), np.nan)
    rsi = np.full(len(close_daily), np.nan)
    
    if len(close_daily) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_daily)):
            avg_gain[i] = (gain[i-1] * 13 + avg_gain[i-1]) / 14
            avg_loss[i] = (loss[i-1] * 13 + avg_loss[i-1]) / 14
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
    
    # Calculate daily average volume for volume confirmation
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high_20 = np.full(n, np.nan)
    lowest_low_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            highest_high_20[i] = np.max(high[i-20:i])
            lowest_low_20[i] = np.min(low[i-20:i])
    
    # Align daily indicators to 4h timeframe
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average of daily volume
        vol_confirm = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            vol_confirm = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with RSI filter and volume confirmation
            # Avoid overextended conditions: RSI between 30 and 70
            rsi_mid = (rsi_daily_aligned[i] >= 30) & (rsi_daily_aligned[i] <= 70)
            
            # Long when price breaks above Donchian upper band
            long_condition = (
                close[i] > highest_high_20[i] and   # breakout above upper band
                rsi_mid and                         # not overextended
                vol_confirm                         # volume confirmation
            )
            
            # Short when price breaks below Donchian lower band
            short_condition = (
                close[i] < lowest_low_20[i] and     # breakdown below lower band
                rsi_mid and                         # not overextended
                vol_confirm                         # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian middle or RSI overextended
            donchian_mid = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < donchian_mid or rsi_daily_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian middle or RSI overextended
            donchian_mid = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > donchian_mid or rsi_daily_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals