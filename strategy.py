#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation (>1.5x avg volume).
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d EMA34 > EMA89 AND volume > 1.5x avg.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d EMA34 < EMA89 AND volume > 1.5x avg.
# Uses ATR(20) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Works in bull markets via trend-following longs and in bear markets via trend-following shorts.
# Elder Ray provides early momentum signals while 1d EMA filter ensures higher timeframe alignment.

name = "6h_ElderRay_BullBearPower_1dEMATrend_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA13 for Elder Ray (on close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate smoothed Elder Ray for momentum (3-period EMA of raw values)
    bull_power_s = pd.Series(bull_power)
    bear_power_s = pd.Series(bear_power)
    bull_power_smooth = bull_power_s.ewm(span=3, adjust=False, min_periods=3).mean().values
    bear_power_smooth = bear_power_s.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = close_1d_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 6h timeframe (wait for daily bar to close)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema89_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i]) or
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power rising (less negative) AND 1d EMA34 > EMA89 AND volume > 1.5x average
            if (bull_power_smooth[i] > 0 and 
                bear_power_smooth[i] > bear_power_smooth[i-1] and  # Bear Power rising
                ema34_1d_aligned[i] > ema89_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Bear Power < 0 AND Bull Power falling (less positive) AND 1d EMA34 < EMA89 AND volume > 1.5x average
            elif (bear_power_smooth[i] < 0 and 
                  bull_power_smooth[i] < bull_power_smooth[i-1] and  # Bull Power falling
                  ema34_1d_aligned[i] < ema89_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals