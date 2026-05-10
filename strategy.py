#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1/S1) on daily chart provide high-probability support/resistance.
Long when price breaks above R1 in uptrend (price > 1d EMA34) with volume spike.
Short when price breaks below S1 in downtrend (price < 1d EMA34) with volume spike.
Exit when price returns to the daily pivot point (PP).
Uses 1d trend and volume confirmation to filter false breakouts.
Designed for 4h timeframe with 20-30 trades/year to minimize fee drag.
Works in bull markets by buying breakouts above resistance in uptrends.
Works in bear markets by selling breakdowns below support in downtrends.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily data for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # PP = (High + Low + Close) / 3
    rang = daily_high - daily_low
    r1 = daily_close + rang * 1.1 / 12
    s1 = daily_close - rang * 1.1 / 12
    pp = (daily_high + daily_low + daily_close) / 3
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 1d EMA34 for trend filter
    ema34_1d = np.full(len(daily_close), np.nan)
    if len(daily_close) >= 34:
        ema34_1d[33] = np.mean(daily_close[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(daily_close)):
            ema34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation spike
    vol_sma20_1d = np.full(len(daily_volume), np.nan)
    if len(daily_volume) >= 20:
        vol_sma20_1d[19] = np.mean(daily_volume[:20])
        for i in range(20, len(daily_volume)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + daily_volume[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for all indicators
    start_idx = 34  # Need at least 34 days for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled)
        # 1 day = 6 * 4h bars, so scale daily average by 1/6 for per-4h comparison
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        if position == 0:
            # Long: Break above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to pivot point (mean reversion to daily mean)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to pivot point
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals