#!/usr/bin/env python3
# 1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeConfirmation
# Hypothesis: 1-hour strategy using 4-hour trend filter and 1-day volume confirmation.
# Uses Camarilla R1/S1 levels from 1-hour pivot for breakout/breakdown.
# Trend filter: price above/below 4-hour EMA50 (medium-term trend).
# Volume confirmation: 1-day volume > 1.5 * 20-day average (institutional participation).
# Designed for low trade frequency (15-30/year) to avoid fee drag in 1h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (mean reversion from extremes via short entries).

name = "1H_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeConfirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-hour data for pivot calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    rang = prev_high - prev_low
    R1 = prev_close + 1.1 * rang * 1.1 / 4
    S1 = prev_close - 1.1 * rang * 1.1 / 4
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1-day volume average for confirmation
    vol_20d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    volume_confirm = volume > (1.5 * vol_20d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup for EMA50
        if (np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume confirmation + price above 4h EMA50 (uptrend)
            if (close[i] > R1[i] and 
                volume_confirm[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume confirmation + price below 4h EMA50 (downtrend)
            elif (close[i] < S1[i] and 
                  volume_confirm[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous 1h H-L range OR closes below 4h EMA50
            if (close[i] < R1[i] and close[i] > S1[i]) or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters previous 1h H-L range OR closes above 4h EMA50
            if (close[i] < R1[i] and close[i] > S1[i]) or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals