#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Support/Resistance Breakout + 1d EMA Trend Filter + Volume Spike
# Uses Camarilla pivot levels from daily data for entry/exit signals, filtered by daily EMA34 trend direction
# and confirmed by volume spikes (>2x 20-period average). Designed to work in both bull and bear markets
# by following the daily trend while using Camarilla levels as dynamic support/resistance.
# Target: 20-50 trades/year.

name = "4h_Camarilla_PivotBreakout_1dEMA34_VolumeSpike"
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
    
    # Get daily data for Camarilla pivots, EMA trend, and volume average
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
    
    # Calculate Camarilla pivot levels for each daily bar
    # Based on previous day's OHLC
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla formulas
    R4 = close_prev + (high_prev - low_prev) * 1.1 / 2
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    R2 = close_prev + (high_prev - low_prev) * 1.1 / 6
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    S2 = close_prev - (high_prev - low_prev) * 1.1 / 6
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    S4 = close_prev - (high_prev - low_prev) * 1.1 / 2
    
    # Calculate daily volume average for volume spike detection
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    R2_aligned = align_htf_to_ltf(prices, df_daily, R2)
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    R4_aligned = align_htf_to_ltf(prices, df_daily, R4)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    S2_aligned = align_htf_to_ltf(prices, df_daily, S2)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    S4_aligned = align_htf_to_ltf(prices, df_daily, S4)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla levels in direction of daily EMA trend
            # Long when price breaks above R1 with volume spike in bullish trend
            long_condition = (
                close[i] > R1_aligned[i] and     # price breaks above R1 resistance
                close[i] > ema34_daily_aligned[i] and  # price above EMA34 (bullish bias)
                vol_spike                        # volume spike for confirmation
            )
            
            # Short when price breaks below S1 with volume spike in bearish trend
            short_condition = (
                close[i] < S1_aligned[i] and     # price breaks below S1 support
                close[i] < ema34_daily_aligned[i] and  # price below EMA34 (bearish bias)
                vol_spike                        # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R1 or trend reverses
            if close[i] < R1_aligned[i] or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S1 or trend reverses
            if close[i] > S1_aligned[i] or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals