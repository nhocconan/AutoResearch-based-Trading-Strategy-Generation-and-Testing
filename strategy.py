#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# Elder Ray measures bullish/bearish power relative to EMA. Combined with higher timeframe
# trend and volume spikes, it captures momentum shifts while filtering weak moves.
# Works in bull/bear by requiring alignment with 12h EMA trend.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 13-period EMA on 12h close for Elder Ray
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA) and Bear Power (Low - EMA) on 12h
    bull_power_12h = high_12h - ema_13_12h
    bear_power_12h = low_12h - ema_13_12h
    
    # Align Elder Ray components to 6h timeframe (wait for 12h close)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # 12h EMA trend filter (21-period)
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Volume filter: volume > 1.5 x 12-period average (3 days of 6h bars)
    vol_ma_12 = np.full(n, np.nan)
    for i in range(11, n):
        vol_ma_12[i] = np.mean(volume[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h data (1 bar), EMA13 (13), EMA21 (21), volume MA (12)
    start_idx = max(1, 13, 21, 12)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(ema_21_aligned[i]) or
            np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_12[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_21_aligned[i]
        bearish_trend = price < ema_21_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, with volume and bullish trend
            # Bear Power < 0 indicates bears weakening
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and
                bear_power_aligned[i] < 0 and
                vol_filter and bullish_trend):
                signals[i] = size
                position = 1
            # Short: Bear Power < 0 and falling, with volume and bearish trend
            # Bull Power > 0 indicates bulls weakening
            elif (bear_power_aligned[i] < 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and
                  bull_power_aligned[i] > 0 and
                  vol_filter and bearish_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or trend turns bearish
            if bull_power_aligned[i] <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power turns positive or trend turns bullish
            if bear_power_aligned[i] >= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0