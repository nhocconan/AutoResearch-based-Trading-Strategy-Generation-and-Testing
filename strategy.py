#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Elder Ray = Bull Power (high - EMA) and Bear Power (low - EMA)
# Long: Bull Power > 0, Bear Power < 0, price > 1d EMA50, volume spike
# Short: Bear Power < 0, Bull Power > 0, price < 1d EMA50, volume spike
# Uses 1d trend filter to align with higher timeframe direction, reducing whipsaw.
# Target: 15-25 trades/year per symbol (60-100 total) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray components on 1d data
    # Bull Power = High - EMA
    # Bear Power = Low - EMA
    bull_power = high_1d - ema_50_1d
    bear_power = low_1d - ema_50_1d
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 6-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > 1d EMA50, volume spike
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, price < 1d EMA50, volume spike
            elif (bear_power_aligned[i] < 0 and bull_power_aligned[i] > 0 and 
                  close[i] < ema_50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray signals reverse or price crosses EMA
            if position == 1:
                if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0 or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Volume_Session"
timeframe = "6h"
leverage = 1.0