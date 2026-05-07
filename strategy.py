#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R1/S1) as breakout triggers on 12h timeframe.
# Requires price to break above R1 (long) or below S1 (short) with volume confirmation and 1d trend filter.
# Trend filter: 1d EMA34 slope (rising for long, falling for short). Avoids counter-trend trades.
# Volume: current volume > 1.5x 20-period average to confirm breakout strength.
# Stops: exit when price re-enters the Camarilla range (between S1 and R1) or trend reverses.
# Designed for low frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels (using prior close, high, low)
    # Shift by 1 to use only completed 1d bar
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no prior
    
    # Camarilla R1 and S1
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # 1d EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA slope: positive if current > previous
    ema34_slope = np.diff(ema34, prepend=np.nan)
    
    # Align to 12h timeframe (wait for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    
    # Volume confirmation: current vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA and roll
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_slope_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, rising EMA trend, volume confirmation
            if (close[i] > R1_aligned[i] and 
                ema34_slope_aligned[i] > 0 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, falling EMA trend, volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  ema34_slope_aligned[i] < 0 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters range (below R1) or trend turns down
            if (close[i] < R1_aligned[i] or 
                ema34_slope_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters range (above S1) or trend turns up
            if (close[i] > S1_aligned[i] or 
                ema34_slope_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals