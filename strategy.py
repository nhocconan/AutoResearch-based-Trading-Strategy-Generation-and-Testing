#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 1d trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low on 6h.
# Long when Bull Power > 0 AND Bear Power rising (improving) AND price > 1d EMA50 AND volume > 1.5x 20-bar MA.
# Short when Bear Power < 0 AND Bull Power falling (deteriorating) AND price < 1d EMA50 AND volume > 1.5x 20-bar MA.
# Uses 1d EMA for trend filter to avoid counter-trend trades in strong trends.
# Volume spike confirms conviction. Designed for 6h timeframe to capture multi-session moves.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Elder Ray: Bull Power and Bear Power using EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Slope of Bear Power (for improving/deteriorating condition)
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    # Slope of Bull Power
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bull Power rising AND price > 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_slope[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bear Power falling AND price < 1d EMA50 AND volume spike
            elif (bear_power[i] > 0 and 
                  bear_power_slope[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power deteriorates or price crosses EMA50
            if position == 1:
                if (bull_power[i] <= 0 or bull_power_slope[i] <= 0 or close[i] < ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (bear_power[i] <= 0 or bear_power_slope[i] >= 0 or close[i] > ema50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0