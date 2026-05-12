#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe (R1/S1) as breakout levels.
# Long when price breaks above R1 with volume confirmation and daily trend up.
# Short when price breaks below S1 with volume confirmation and daily trend down.
# Uses daily EMA34 for trend filter and volume spike (volume > 1.5x 20-period average).
# Designed for low frequency (20-40 trades/year) to avoid fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns R1, S1, R2, S2, R3, S3, R4, S4, PP (pivot point).
    """
    typical = (high + low + close) / 3
    range_ = high - low
    
    # Avoid division by zero
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    # Calculate pivot point
    pp = typical
    
    # Calculate levels
    r1 = pp + (range_ * 1.1 / 12)
    s1 = pp - (range_ * 1.1 / 12)
    r2 = pp + (range_ * 1.1 / 6)
    s2 = pp - (range_ * 1.1 / 6)
    r3 = pp + (range_ * 1.1 / 4)
    s3 = pp - (range_ * 1.1 / 4)
    r4 = pp + (range_ * 1.1 / 2)
    s4 = pp - (range_ * 1.1 / 2)
    
    return r1, s1, r2, s2, r3, s3, r4, s4, pp

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    r1_1d, s1_1d, _, _, _, _, _, _, pp_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily Camarilla levels and EMA to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume spike: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_above_r1 = close[i] > r1_1d_aligned[i]
        breakdown_below_s1 = close[i] < s1_1d_aligned[i]
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: Breakout above R1 + volume spike + daily trend up
            if breakout_above_r1 and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 + volume spike + daily trend down
            elif breakdown_below_s1 and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Breakdown below S1 OR trend turns down
            if breakdown_below_s1 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Breakout above R1 OR trend turns up
            if breakout_above_r1 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals