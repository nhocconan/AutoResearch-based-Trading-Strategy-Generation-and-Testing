#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bullish/bearish power relative to EMA13. In 6h timeframe:
# - Bull Power = High - EMA13 (measures buying strength)
# - Bear Power = Low - EMA13 (measures selling strength)
# Long when: Bull Power > 0 (bullish momentum) AND Bear Power rising from negative (selling exhaustion)
# Short when: Bear Power < 0 (bearish momentum) AND Bull Power falling from positive (buying exhaustion)
# 1d EMA50 provides higher-timeframe trend filter to avoid counter-trend whipsaws
# Volume confirmation (1.5x 20-period EMA) ensures institutional participation
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25)
# Works in bull markets by buying strength on pullbacks and in bear markets by selling weakness on rallies

name = "6h_ElderRay_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate smoothed Elder Ray (13-period EMA of raw values) to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume confirmation: 1.5x 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND Bear Power rising from negative (selling exhaustion) 
            #         AND price above 1d EMA50 (uptrend filter) AND volume confirmation
            if (bull_power_smooth[i] > 0.0 and bear_power_smooth[i] > bear_power_smooth[i-1] and 
                close[i] > ema_50_1d_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND Bull Power falling from positive (buying exhaustion)
            #        AND price below 1d EMA50 (downtrend filter) AND volume confirmation
            elif (bear_power_smooth[i] < 0.0 and bull_power_smooth[i] < bull_power_smooth[i-1] and 
                  close[i] < ema_50_1d_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative (loss of buying pressure) OR price below 1d EMA50 (trend change)
            if bull_power_smooth[i] <= 0.0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive (loss of selling pressure) OR price above 1d EMA50 (trend change)
            if bear_power_smooth[i] >= 0.0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals