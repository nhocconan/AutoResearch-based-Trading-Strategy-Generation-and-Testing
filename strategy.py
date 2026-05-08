#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA34 Trend + Volume Spike
# Uses daily EMA34 for trend direction, Williams %R(14) for mean-reversion entries,
# and volume spike (>1.5x 20-period average) for confirmation. Designed to capture
# reversals in both bull and bear markets by trading pullbacks to the trend with
# momentum confirmation. Target: 25-40 trades/year.

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
    
    # Get daily data for EMA trend filter and Williams %R
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
    
    # Calculate daily Williams %R (14-period)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    williams_r = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 14:
        for i in range(13, len(close_daily)):
            highest_high = np.max(high_daily[i-13:i+1])
            lowest_low = np.min(low_daily[i-13:i+1])
            if highest_high != lowest_low:
                williams_r[i] = -100 * (highest_high - close_daily[i]) / (highest_high - lowest_low)
            else:
                williams_r[i] = -50  # neutral when no range
    
    # Calculate daily volume average for volume spike
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 4h volume > 1.5x 20-period average of daily volume
        vol_spike = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R oversold/overbought with trend alignment
            # Williams %R < -80 = oversold, > -20 = overbought
            williams_oversold = williams_r_aligned[i] < -80
            williams_overbought = williams_r_aligned[i] > -20
            
            # Long when oversold in uptrend (price above EMA34)
            long_condition = (
                williams_oversold and           # oversold condition
                close[i] > ema34_daily_aligned[i] and   # price above EMA34 (uptrend)
                vol_spike                       # volume spike for confirmation
            )
            
            # Short when overbought in downtrend (price below EMA34)
            short_condition = (
                williams_overbought and         # overbought condition
                close[i] < ema34_daily_aligned[i] and   # price below EMA34 (downtrend)
                vol_spike                       # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend breaks
            if williams_r_aligned[i] > -50 or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend breaks
            if williams_r_aligned[i] < -50 or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals