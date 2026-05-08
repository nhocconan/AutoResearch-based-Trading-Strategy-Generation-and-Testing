#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Keltner Channel Breakout with Weekly Trend Filter and Volume Confirmation
# Uses weekly EMA20 trend for directional bias, Keltner Channel (ATR-based) for breakout detection,
# and volume spike (>2x average) for entry confirmation. Designed to capture trending moves
# in both bull and bear markets by following the weekly trend while avoiding false breakouts.
# Target: 15-35 trades/year per symbol.

name = "6h_Keltner_WeeklyEMA20_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema20_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 20:
        ema20_weekly[19] = np.mean(close_weekly[:20])
        for i in range(20, len(close_weekly)):
            ema20_weekly[i] = (close_weekly[i] * 2 + ema20_weekly[i-1] * 18) / 20
    
    # Get daily data for ATR (used in Keltner Channel)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily ATR(10) for Keltner Channel
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr = np.maximum(high_daily[1:] - low_daily[1:], 
                    np.maximum(np.abs(high_daily[1:] - close_daily[:-1]),
                               np.abs(low_daily[1:] - close_daily[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    atr_10_daily = np.full(len(tr), np.nan)
    if len(tr) >= 10:
        atr_10_daily[9] = np.nanmean(tr[:10])
        for i in range(10, len(tr)):
            if np.isnan(atr_10_daily[i-1]):
                atr_10_daily[i] = np.nanmean(tr[i-9:i+1])
            else:
                atr_10_daily[i] = (atr_10_daily[i-1] * 9 + tr[i]) / 10
    
    # Calculate daily Keltner Channel components (20-period EMA of close ± 2*ATR)
    ema_kc_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 20:
        ema_kc_daily[19] = np.mean(close_daily[:20])
        for i in range(20, len(close_daily)):
            ema_kc_daily[i] = (close_daily[i] * 2 + ema_kc_daily[i-1] * 18) / 20
    
    upper_keltner = np.full(len(close_daily), np.nan)
    lower_keltner = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 20:
        for i in range(20, len(close_daily)):
            if not np.isnan(ema_kc_daily[i]) and not np.isnan(atr_10_daily[i]):
                upper_keltner[i] = ema_kc_daily[i] + 2.0 * atr_10_daily[i]
                lower_keltner[i] = ema_kc_daily[i] - 2.0 * atr_10_daily[i]
    
    # Calculate daily average volume for volume spike filter
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly and daily indicators to 6h timeframe
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    ema_kc_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_kc_daily)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_daily, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_daily, lower_keltner)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema20_weekly_aligned[i]) or np.isnan(ema_kc_daily_aligned[i]) or
            np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 6h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout of Keltner Channel in direction of weekly trend
            # Weekly trend: price > EMA20 = bullish, price < EMA20 = bearish
            weekly_bullish = close[i] > ema20_weekly_aligned[i]
            weekly_bearish = close[i] < ema20_weekly_aligned[i]
            
            # Long when price breaks above upper Keltner in bullish weekly trend
            long_condition = (
                close[i] > upper_keltner_aligned[i] and   # breakout above Keltner upper band
                weekly_bullish and                        # weekly trend is bullish
                vol_spike                                 # volume confirmation
            )
            
            # Short when price breaks below lower Keltner in bearish weekly trend
            short_condition = (
                close[i] < lower_keltner_aligned[i] and   # breakout below Keltner lower band
                weekly_bearish and                        # weekly trend is bearish
                vol_spike                                 # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA-based middle of Keltner Channel
            if close[i] < ema_kc_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above EMA-based middle of Keltner Channel
            if close[i] > ema_kc_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals