#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (EMA13 from 6h data).
# Enter long when Bull Power > 0 and rising (current > previous) AND 12h close > EMA50.
# Enter short when Bear Power > 0 and rising AND 12h close < EMA50.
# Volume filter: current volume > 1.3x 20-period average.
# Exit when power becomes negative or trend reverses.
# Designed for ~15-25 trades/year per side with strict conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate EMA50 for 12h trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period volume MA and 13-period EMA
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filters from 12h EMA50
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Elder Ray rising conditions
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        bear_power_rising = i > 0 and bear_power[i] > bear_power[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 and rising + volume + bullish 12h trend
            if bull_power[i] > 0 and bull_power_rising and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: Bear Power > 0 and rising + volume + bearish 12h trend
            elif bear_power[i] > 0 and bear_power_rising and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 or trend turns bearish
            if bull_power[i] <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power <= 0 or trend turns bullish
            if bear_power[i] <= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0