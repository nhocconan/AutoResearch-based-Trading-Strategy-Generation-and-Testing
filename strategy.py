#!/usr/bin/env python3
# 12H_DonchianBreakout_VolumeConfirmed_TrendFilter
# Hypothesis: 12h Donchian channel breakouts with volume confirmation (1.5x 20-period average) and trend filter (price > 12h EMA 50) capture strong directional moves. Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes via short entries). Target: 15-30 trades/year (60-120 total over 4 years) to stay within 12h limits.

name = "12H_DonchianBreakout_VolumeConfirmed_TrendFilter"
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
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper = high_roll
    lower = low_roll
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # 1d data for trend filter: EMA 50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_ema = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(daily_ema_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume confirmation + price above daily EMA (uptrend)
            if close[i] > upper[i] and volume_confirm[i] and close[i] > daily_ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume confirmation + price below daily EMA (downtrend)
            elif close[i] < lower[i] and volume_confirm[i] and close[i] < daily_ema_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (between lower and upper) OR closes below daily EMA (trend change)
            if close[i] < upper[i] and close[i] > lower[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel OR closes above daily EMA (trend change)
            if close[i] < upper[i] and close[i] > lower[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > daily_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals