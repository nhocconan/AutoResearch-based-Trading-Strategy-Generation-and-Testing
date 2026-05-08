#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA Trend Filter + Volume Spike
# Uses Williams %R for mean-reversion entries during pullbacks in the direction of the daily trend.
# Volume spike confirms institutional participation. Designed to work in both bull and bear markets
# by following the daily EMA34 trend while entering on overextended pullbacks. Target: 25-50 trades/year.

name = "4h_WilliamsR_1dEMA34_VolumeSpike"
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
    
    # Get daily data for EMA trend filter and volume average
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
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Calculate Williams %R (14-period) on 4h data
    williams_r = np.full(n, np.nan)
    if n >= 14:
        for i in range(14, n):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high > lowest_low:
                williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                williams_r[i] = -50  # neutral when no range
    
    # Align daily indicators to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
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
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 20-period average of daily volume
        vol_spike = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R overextended in direction of daily trend
            # Williams %R < -80 = oversold (long opportunity)
            # Williams %R > -20 = overbought (short opportunity)
            oversold = williams_r[i] < -80
            overbought = williams_r[i] > -20
            
            # Long when oversold and price above daily EMA34 (bullish trend)
            long_condition = (
                oversold and
                close[i] > ema34_daily_aligned[i] and
                vol_spike
            )
            
            # Short when overbought and price below daily EMA34 (bearish trend)
            short_condition = (
                overbought and
                close[i] < ema34_daily_aligned[i] and
                vol_spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend changes
            if williams_r[i] > -50 or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend changes
            if williams_r[i] < -50 or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals