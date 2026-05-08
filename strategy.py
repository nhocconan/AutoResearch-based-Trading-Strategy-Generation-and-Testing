#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + 1d EMA Trend Filter + Volume Breakout
# Uses daily EMA34 trend direction for bias, choppiness index to filter trending markets,
# and volume breakout (>2x average) for entry timing. Designed to work in both bull and bear
# markets by following the daily trend while avoiding choppy conditions. Target: 20-50 trades/year.

name = "4h_Choppiness_1dEMA34_VolumeBreakout"
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
    
    # Get daily data for EMA trend filter and choppiness
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate daily choppiness index (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    atr_14_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        tr = np.maximum(high_daily[1:] - low_daily[1:], 
                        np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                                   np.abs(low_daily[1:] - close_daily[:-1])))
        tr = np.concatenate([[np.nan], tr])
        for i in range(14, len(tr)):
            if np.isnan(atr_14_daily[i-1]):
                atr_14_daily[i] = np.nanmean(tr[i-13:i+1])
            else:
                atr_14_daily[i] = (atr_14_daily[i-1] * 13 + tr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = np.full(len(close_daily), np.nan)
    lowest_low_14 = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(14, len(close_daily)):
            highest_high_14[i] = np.max(high_daily[i-13:i+1])
            lowest_low_14[i] = np.min(low_daily[i-13:i+1])
    
    # Choppiness Index: CI = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(14, len(close_daily)):
            if not np.isnan(atr_14_daily[i]) and not np.isnan(highest_high_14[i]) and not np.isnan(lowest_low_14[i]):
                if highest_high_14[i] > lowest_low_14[i]:
                    sum_atr = np.nansum(atr_14_daily[i-13:i+1])
                    chop_daily[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high_14[i] - lowest_low_14[i]) + 1e-10)
                else:
                    chop_daily[i] = 50  # neutral when no range
    
    # Calculate daily volume average for volume breakout
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    chop_daily_aligned = align_htf_to_ltf(prices, df_daily, chop_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(chop_daily_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current 4h volume > 2x 20-period average of daily volume
        # Find current daily bar's volume
        vol_breakout = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_4h_current = volume[i]
                vol_breakout = vol_4h_current > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: follow daily EMA trend in non-choppy market with volume breakout
            # Choppiness < 38.2 indicates trending market (good for trend following)
            trending_market = chop_daily_aligned[i] < 38.2
            
            # Long when price above daily EMA34 in bullish trend
            long_condition = (
                close[i] > ema34_daily_aligned[i] and   # price above EMA34 (bullish bias)
                trending_market and                     # trending market (not choppy)
                vol_breakout                            # volume breakout for entry
            )
            
            # Short when price below daily EMA34 in bearish trend
            short_condition = (
                close[i] < ema34_daily_aligned[i] and   # price below EMA34 (bearish bias)
                trending_market and                     # trending market (not choppy)
                vol_breakout                            # volume breakout for entry
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA34 or market becomes choppy
            if close[i] < ema34_daily_aligned[i] or chop_daily_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above EMA34 or market becomes choppy
            if close[i] > ema34_daily_aligned[i] or chop_daily_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals