#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Index with 1-week trend filter and 1-day volume confirmation.
# Long when: Bull Power > 0, Bear Power < 0, EMA13(1w) > EMA34(1w), volume > 1.5x 20-period average
# Short when: Bear Power < 0, Bull Power < 0, EMA13(1w) < EMA34(1w), volume > 1.5x 20-period average
# Exit when Bull Power and Bear Power have same sign (both positive or both negative).
# Elder Ray measures bull/bear power relative to EMA, working in both bull and bear markets.
# Target: 15-25 trades/year per symbol. Uses EMA smoothing for trend and power calculation.
name = "12h_ElderRay_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 and EMA34 on weekly data
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema13_1w = ema(close_1w, 13)
    ema34_1w = ema(close_1w, 34)
    
    # Align weekly EMAs to 12h timeframe
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 13-period EMA for 12h data (for Elder Ray)
    ema13_12h = ema(close, 13)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_12h
    bear_power = low - ema13_12h
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema13_1w_val = ema13_1w_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, weekly uptrend, volume confirmation
            if bull > 0 and bear < 0 and ema13_1w_val > ema34_1w_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0, Bull Power < 0, weekly downtrend, volume confirmation
            elif bear < 0 and bull < 0 and ema13_1w_val < ema34_1w_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power and Bear Power both positive (bull exhaustion) or both negative (trend change)
            if (bull > 0 and bear > 0) or (bull < 0 and bear < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power and Bear Power both positive (trend change) or both negative (bear exhaustion)
            if (bull > 0 and bear > 0) or (bull < 0 and bear < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals