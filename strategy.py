#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses weekly Donchian channels for trend direction and 6h Donchian breakouts for entry.
# Long when 6h price breaks above 6h Donchian(20) high and weekly trend is up.
# Short when 6h price breaks below 6h Donchian(20) low and weekly trend is down.
# Weekly trend defined as price above/below weekly EMA(20).
# Volume confirmation reduces false breakouts.
# Designed for 6h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via weekly trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = close_1w > ema_20_1w
    weekly_downtrend = close_1w < ema_20_1w
    
    # Align weekly trend to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Calculate 6h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Moderate threshold for balance
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 6h breakout above Donchian high + weekly uptrend + volume spike
            if (close[i] > high_roll[i] and 
                weekly_uptrend_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: 6h breakdown below Donchian low + weekly downtrend + volume spike
            elif (close[i] < low_roll[i] and 
                  weekly_downtrend_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit on opposite Donchian touch or trend reversal
            if position == 1:
                # Exit on Donchian low touch or weekly trend turns down
                if (close[i] < low_roll[i] or weekly_downtrend_aligned[i] > 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian high touch or weekly trend turns up
                if (close[i] > high_roll[i] or weekly_uptrend_aligned[i] > 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0