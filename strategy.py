#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA200 trend + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: bull power = high - EMA13, bear power = low - EMA13
# 1d EMA200 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 20-period average) ensures breakout conviction
# Works in bull/bear: EMA200 trend filter aligns with dominant market direction
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_elder_ray_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend direction
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: bull power and bear power
    bull_power = high - ema_13  # Bull power = high - EMA13
    bear_power = low - ema_13   # Bear power = low - EMA13
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: bear power > 0 (bulls losing control) OR price < 1d EMA200 (trend change)
            if bear_power[i] > 0 or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bull power < 0 (bears losing control) OR price > 1d EMA200 (trend change)
            if bull_power[i] < 0 or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Elder Ray + EMA200 trend filter
            if volume_confirmed:
                # Long entry: bull power > 0 AND price > 1d EMA200 (strong bulls + uptrend)
                if bull_power[i] > 0 and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: bear power < 0 AND price < 1d EMA200 (strong bears + downtrend)
                elif bear_power[i] < 0 and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals