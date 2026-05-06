#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long when Bull Power > 0 AND Bear Power increasing (less negative) AND price > 1d EMA50 AND volume spike
# Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND price < 1d EMA50 AND volume spike
# Uses 13-period EMA for Elder Ray (standard) and 50-period for trend filter
# Volume spike (>1.5x 20-bar average) confirms participation
# Designed for 6h timeframe to capture medium-term trends with lower frequency
# Works in bull via Bull Power strength, in bear via Bear Power extremes
# Target: 60-120 total trades over 4 years (15-30/year)

name = "6h_ElderRay_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA13 for Elder Ray (using close)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Calculate Elder Ray momentum (change in power)
    bull_power_momentum = bull_power - np.roll(bull_power, 1)
    bear_power_momentum = bear_power - np.roll(bear_power, 1)
    # Set first value to 0
    bull_power_momentum[0] = 0
    bear_power_momentum[0] = 0
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_momentum[i]) or 
            np.isnan(bear_power_momentum[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND increasing AND price above 1d EMA50 AND volume spike
            if (bull_power[i] > 0 and bull_power_momentum[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND decreasing (more negative) AND price below 1d EMA50 AND volume spike
            elif (bear_power[i] < 0 and bear_power_momentum[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR Bear Power becomes more negative than -0.5
            if bull_power[i] <= 0 or bear_power[i] < -0.5 * np.std(bear_power[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR Bull Power becomes less than 0.5
            if bear_power[i] >= 0 or bull_power[i] < 0.5 * np.std(bull_power[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals