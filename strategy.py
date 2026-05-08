#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w Trend Filter + Volume Breakout
# Uses weekly EMA50 trend direction for bias, Williams Alligator (13,8,5 SMAs) to filter trending markets,
# and volume breakout (>1.5x average) for entry timing. Designed to work in both bull and bear
# markets by following the weekly trend while avoiding choppy conditions. Target: 10-25 trades/year.

name = "1d_WilliamsAlligator_1wEMA50_VolumeBreakout"
timeframe = "1d"
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
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Get daily data for Williams Alligator and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator: Jaw (13-period SMA), Teeth (8-period SMA), Lips (5-period SMA)
    close_daily = df_daily['close'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Jaw: 13-period SMA
    jaw = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 13:
        for i in range(13, len(close_daily)):
            jaw[i] = np.mean(close_daily[i-12:i+1])
    
    # Teeth: 8-period SMA
    teeth = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 8:
        for i in range(8, len(close_daily)):
            teeth[i] = np.mean(close_daily[i-7:i+1])
    
    # Lips: 5-period SMA
    lips = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 5:
        for i in range(5, len(close_daily)):
            lips[i] = np.mean(close_daily[i-4:i+1])
    
    # Williams Alligator signals: 
    # Alligator sleeping (all lines intertwined) -> no trend
    # Alligator awake (lines separated) -> trend present
    # Specifically: Lips > Teeth > Jaw = bullish alignment
    #             Lips < Teeth < Jaw = bearish alignment
    
    # Calculate daily volume average for volume breakout
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly indicators to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Align daily indicators to daily timeframe (no alignment needed for same timeframe,
    # but we keep the pattern for consistency and to handle any potential timeframe issues)
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current daily volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        # Williams Alligator alignment
        # Bullish: Lips > Teeth > Jaw
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish: Lips < Teeth < Jaw
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Look for entry: follow weekly EMA trend with Alligator alignment and volume breakout
            # Long when price above weekly EMA50 in bullish trend with Alligator bullish alignment
            long_condition = (
                close[i] > ema50_weekly_aligned[i] and   # price above weekly EMA50 (bullish bias)
                bullish_alignment and                    # Alligator bullish alignment
                vol_breakout                             # volume breakout for entry
            )
            
            # Short when price below weekly EMA50 in bearish trend with Alligator bearish alignment
            short_condition = (
                close[i] < ema50_weekly_aligned[i] and   # price below weekly EMA50 (bearish bias)
                bearish_alignment and                    # Alligator bearish alignment
                vol_breakout                             # volume breakout for entry
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below weekly EMA50 or Alligator turns bearish
            if close[i] < ema50_weekly_aligned[i] or bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly EMA50 or Alligator turns bullish
            if close[i] > ema50_weekly_aligned[i] or bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals