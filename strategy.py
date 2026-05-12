#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
# Hypothesis: Use 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume spike (2x 20-period MA).
# Long when price breaks above R1 with price > 4h EMA50 and volume > 2x MA.
# Short when price breaks below S1 with price < 4h EMA50 and volume > 2x MA.
# Exit when price reverses back through the Camarilla pivot point.
# Session filter: 08-20 UTC to avoid low-volume hours.
# Designed for 1h timeframe to capture intraday breaks with trend and volume confirmation.
# Targets 15-30 trades/year to minimize fee drag while maintaining edge in bull/bear markets.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate previous day's OHLC for Camarilla levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    pivot = (prev_high + prev_low + prev_close) / 3  # Pivot point for exit
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pivot[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with price > 4h EMA50 and volume > 2x MA
            if close[i] > r1[i] and close[i] > ema50_4h_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with price < 4h EMA50 and volume > 2x MA
            elif close[i] < s1[i] and close[i] < ema50_4h_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back below pivot point
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price moves back above pivot point
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals