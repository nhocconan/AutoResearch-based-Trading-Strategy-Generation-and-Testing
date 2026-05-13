#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 1w EMA34 trend filter and volume confirmation (>1.5x avg volume).
# Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close). 
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1w EMA34 > EMA89 AND volume > 1.5x avg.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1w EMA34 < EMA89 AND volume > 1.5x avg.
# Uses ATR(20) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Elder Ray measures bull/bear strength relative to EMA13, effective in both trending and ranging markets.
# Weekly EMA trend filter ensures we trade with higher timeframe momentum, reducing whipsaw.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_BullBearPower_1wEMATrend_VolumeConfirm_v1"
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
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 and EMA89 for trend filter
    close_1w_s = pd.Series(close_1w)
    ema34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1w = close_1w_s.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMAs to 6h timeframe (wait for weekly bar to close)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema89_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema89_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power rising (less negative than previous) AND 1w EMA34 > EMA89 AND volume > 1.5x average
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative)
                ema34_1w_aligned[i] > ema89_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Bear Power < 0 AND Bull Power falling (less positive than previous) AND 1w EMA34 < EMA89 AND volume > 1.5x average
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive)
                  ema34_1w_aligned[i] < ema89_1w_aligned[i] and 
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