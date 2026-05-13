#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use 4h EMA50 for trend direction and 1d volume spike for confirmation, entering on 1h breaks of daily Camarilla R1/S1. Designed for low trade frequency (<30/year) to avoid fee drag in both bull and bear markets by combining multi-timeframe confluence with volume filter.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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
    
    # Get 1d data for Camarilla levels (based on previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    camarilla_r1[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    camarilla_s1[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d volume confirmation: volume > 2.0x 20-day average
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # LONG: Price breaks above R1 with 1d volume spike in 4h uptrend (close > EMA50)
        if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
           high[i] > camarilla_r1_aligned[i] and volume_spike_1d_aligned[i] and close[i] > ema_50_4h_aligned[i]:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # SHORT: Price breaks below S1 with 1d volume spike in 4h downtrend (close < EMA50)
        elif camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
             low[i] < camarilla_s1_aligned[i] and volume_spike_1d_aligned[i] and close[i] < ema_50_4h_aligned[i]:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # EXIT: Price returns to the opposite Camarilla level or trend weakens
        elif position == 1 and (low[i] < camarilla_s1_aligned[i] or close[i] < ema_50_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (high[i] > camarilla_r1_aligned[i] or close[i] > ema_50_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals