#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Trend Filter + Volume Spike
# Uses 6h EMA(13) as pivot. Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 and rising, Bear Power > 0 and falling, 1d EMA(50) up, volume > 1.5x 20 EMA.
# Short when Bear Power > 0 and rising, Bull Power > 0 and falling, 1d EMA(50) down, volume > 1.5x 20 EMA.
# Designed for low trade frequency (15-25/year) to minimize fee flood and capture momentum with trend alignment.

name = "6h_ElderRay_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # 6h EMA(13) for Elder Ray pivot
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = EMA(13) - Low
    bear_power = ema_13 - low
    
    # Slope of Bull/Bear Power (1-period change)
    bull_power_slope = bull_power[1:] - bull_power[:-1]
    bull_power_slope = np.concatenate([[0], bull_power_slope])
    bear_power_slope = bear_power[1:] - bear_power[:-1]
    bear_power_slope = np.concatenate([[0], bear_power_slope])
    
    # Align 1d trend to 6h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    # Volume confirmation: 6h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for EMA(20) volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: Bull Power > 0 and rising, Bear Power > 0 and falling, 1d uptrend, volume
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and
                bear_power[i] > 0 and bear_power_slope[i] < 0 and
                trend_up_aligned[i] > 0.5 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: Bear Power > 0 and rising, Bull Power > 0 and falling, 1d downtrend, volume
            elif (bear_power[i] > 0 and bear_power_slope[i] > 0 and
                  bull_power[i] > 0 and bull_power_slope[i] < 0 and
                  trend_up_aligned[i] <= 0.5 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or Bear Power rises or trend turns down
            if bull_power[i] <= 0 or bear_power_slope[i] > 0 or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative or Bull Power rises or trend turns up
            if bear_power[i] <= 0 or bull_power_slope[i] > 0 or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals