#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using weekly Donchian(20) breakout with daily EMA50 trend filter and volume confirmation.
Breakouts aligned with daily EMA50 trend (bullish above, bearish below) tend to continue in both bull and bear markets.
Volume > 2.0x average confirms breakout strength. Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
Target: 15-30 trades/year (60-120 over 4 years). Includes ATR-based stoploss to limit drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on weekly data
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    highest_high_weekly = np.full(len(df_weekly), np.nan)
    lowest_low_weekly = np.full(len(df_weekly), np.nan)
    
    for i in range(20, len(df_weekly)):
        highest_high_weekly[i] = np.max(high_weekly[i-20:i])
        lowest_low_weekly[i] = np.min(low_weekly[i-20:i])
    
    # Align weekly Donchian to daily timeframe (waits for weekly bar close)
    highest_high_aligned = align_htf_to_ltf(prices, df_weekly, highest_high_weekly)
    lowest_low_aligned = align_htf_to_ltf(prices, df_weekly, lowest_low_weekly)
    
    # Get daily data for EMA50 trend filter and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close
    close_daily = df_daily['close'].values
    ema_50 = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema_50[49] = np.mean(close_daily[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_daily)):
            ema_50[i] = (close_daily[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align daily EMA50 to daily timeframe (no alignment needed as same timeframe)
    ema_50_aligned = ema_50  # Already on daily timeframe
    
    # Calculate 20-period average daily volume for spike detection
    volume_daily = df_daily['volume'].values
    vol_ma_daily = np.full(len(df_daily), np.nan)
    for i in range(20, len(df_daily)):
        vol_ma_daily[i] = np.mean(volume_daily[i-20:i])
    
    # Align daily volume MA to daily timeframe
    vol_ma_aligned = vol_ma_daily  # Already on daily timeframe
    
    # ATR for stoploss (using daily data)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily_arr = df_daily['close'].values
    tr = np.zeros(len(df_daily))
    atr = np.full(len(df_daily), np.nan)
    for i in range(1, len(df_daily)):
        tr[i] = max(high_daily[i] - low_daily[i], abs(high_daily[i] - close_daily_arr[i-1]), abs(low_daily[i] - close_daily_arr[i-1]))
    
    for i in range(14, len(df_daily)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to daily timeframe
    atr_aligned = atr  # Already on daily timeframe
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 20 for weekly Donchian, 50 for daily EMA50, 20 for volume MA, 14 for ATR
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_aligned[i]) or
            np.isnan(lowest_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        # Determine trend from daily EMA50
        bullish = price > ema_50_aligned[i]
        bearish = price < ema_50_aligned[i]
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high in bullish trend with volume
            if bullish and price > highest_high_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below weekly Donchian low in bearish trend with volume
            elif bearish and price < lowest_low_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low or trend turns bearish or stoploss hit
            if price < lowest_low_aligned[i] or bearish or price < (entry_price - 2.0 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high or trend turns bullish or stoploss hit
            if price > highest_high_aligned[i] or bullish or price > (entry_price + 2.0 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
        
        # Track entry price for stoploss calculation
        if position != 0 and signals[i] != 0:
            if position == 1 and signals[i] == size:
                entry_price = price
            elif position == -1 and signals[i] == -size:
                entry_price = price
    
    return signals

name = "1d_WeeklyDonchian20_DailyEMA50_Volume"
timeframe = "1d"
leverage = 1.0