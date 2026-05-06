#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily pivot points with volume confirmation and volatility filter
# Uses classic pivot point (PP) and support/resistance levels (R1,S1,R2,S2) from previous day
# Price breaking above R2 or below S2 with volume > 1.5x average indicates institutional breakout
# Price rejecting at R1 or S1 with volume confirmation indicates mean reversion opportunity
# Volatility filter (ATR ratio) avoids choppy markets
# Works in both bull/bear markets: breakouts capture trends, reversals capture pullbacks
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_PivotPoint_R2S2_VolumeVolatilityFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Classic pivot point calculation
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Support and Resistance levels
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    # R2 = PP + (High - Low)
    # S2 = PP - (High - Low)
    r1 = (2 * pp) - prev_low
    s1 = (2 * pp) - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    
    # Align daily levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Volatility filter: ATR ratio (current ATR / 20-period average ATR) < 1.5
    # Avoids extremely volatile conditions that cause whipsaws
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_current = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_current).rolling(window=20, min_periods=20).mean().values
    volatility_filter = (atr_current / atr_ma) < 1.5
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(volatility_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume confirmation and acceptable volatility
            if close[i] > r2_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S2 with volume confirmation and acceptable volatility
            elif close[i] < s2_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects S1 with volume confirmation (bounce from support)
            elif close[i] < s1_aligned[i] and close[i] > s1_aligned[i] * 0.998 and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects R1 with volume confirmation (rejection from resistance)
            elif close[i] > r1_aligned[i] and close[i] < r1_aligned[i] * 1.002 and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (failed support) or reaches R1 (take profit)
            if close[i] < s1_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (failed resistance) or reaches S1 (take profit)
            if close[i] > r1_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals