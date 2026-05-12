#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use Camarilla pivot levels from daily data with 12h timeframe for entries.
# Long when price breaks above R1 level with price > daily EMA34 and volume > 2x volume MA.
# Short when price breaks below S1 level with price < daily EMA34 and volume > 2x volume MA.
# Exit when price reverses back to the pivot point (PP).
# Designed to capture intraday swings with multi-timeframe confirmation, works in both bull and bear markets by filtering with daily trend.
# Targets 12-37 trades/year to minimize fee drag on 12h timeframe.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # S1 = PP - (H - L) * 1.1 / 12
    # We need previous day's H, L, C for today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = PP + (prev_high - prev_low) * 1.1 / 12.0
    S1 = PP - (prev_high - prev_low) * 1.1 / 12.0
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(daily_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with price > daily EMA and volume > 2x MA
            if close[i] > R1[i] and close[i] > daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with price < daily EMA and volume > 2x MA
            elif close[i] < S1[i] and close[i] < daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves back to pivot point (PP)
            if close[i] <= PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves back to pivot point (PP)
            if close[i] >= PP[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals