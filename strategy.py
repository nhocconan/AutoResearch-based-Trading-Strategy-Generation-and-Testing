#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Elder Ray uses EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 1d EMA50 uptrend and volume spike.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with 1d EMA50 downtrend and volume spike.
# Designed to capture momentum in both bull and bear markets by measuring bull/bear strength relative to trend.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Calculate 3-period EMA of Bull/Bear Power for slope
    bull_power_series = pd.Series(bull_power)
    bear_power_series = pd.Series(bear_power)
    ema_bull_power = bull_power_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema_bear_power = bear_power_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Slope of Bull/Bear Power (current - previous)
    bull_power_slope = ema_bull_power - np.roll(ema_bull_power, 1)
    bear_power_slope = ema_bear_power - np.roll(ema_bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need EMA13 (13) + EMA of power (3) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema50 = close[i] > ema50_1d_aligned[i]
        price_below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, with 1d EMA50 uptrend and volume
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                bear_power[i] < 0 and price_above_ema50 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0, with 1d EMA50 downtrend and volume
            elif (bear_power[i] < 0 and bear_power_slope[i] < 0 and 
                  bull_power[i] > 0 and price_below_ema50 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR Bear Power turns positive
            if (bull_power[i] <= 0) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive OR Bull Power turns negative
            if (bear_power[i] >= 0) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0