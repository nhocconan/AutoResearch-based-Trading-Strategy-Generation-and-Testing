#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA200 trend filter and volume confirmation (>1.8x avg volume). Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.20.
# Target: 80-120 total trades over 4 years (20-30/year) on 1h timeframe.
# 4h EMA200 ensures we only trade with the higher timeframe trend, reducing counter-trend whipsaw in both bull and bear markets.
# Camarilla R1/S1 breakouts capture strong intraday momentum moves, volume confirmation ensures institutional participation.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Works in bull markets via trend-following breakouts and in bear markets via shorting breakdowns with trend filter.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA200_VolumeConfirm_ATRStop_v1"
timeframe = "1h"
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema200_4h = close_4h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA200 to 1h timeframe (wait for 4h bar to close)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate Camarilla pivot levels for 1h timeframe
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high + low + close) / 3.0
    r1 = close + (high - low) * 1.1 / 12.0
    s1 = close - (high - low) * 1.1 / 12.0
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            # Carry forward tracking values when flat
            if position == 0 and i > 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            elif position == 1 and i > 0:
                highest_since_entry[i] = highest_since_entry[i-1]
            elif position == -1 and i > 0:
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND 4h EMA200 > 0 (rising trend) AND volume > 1.8x average
            if (close[i] > r1[i] and 
                ema200_4h_aligned[i] > np.roll(ema200_4h_aligned, 1)[i] and  # EMA200 rising
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S1 AND 4h EMA200 < 0 (falling trend) AND volume > 1.8x average
            elif (close[i] < s1[i] and 
                  ema200_4h_aligned[i] < np.roll(ema200_4h_aligned, 1)[i] and  # EMA200 falling
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.20
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
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals