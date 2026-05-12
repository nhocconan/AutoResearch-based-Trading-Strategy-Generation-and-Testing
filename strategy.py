#!/usr/bin/env python3
# 12h_Donchian_Breakout_Trend_Volume
# Hypothesis: Donchian(20) breakouts on 12h timeframe with 1d EMA200 trend filter and volume confirmation
# capture strong momentum moves. Works in bull markets via long breakouts above upper band and in bear markets
# via short breakdowns below lower band. Volume ensures breakout validity, trend filter avoids counter-trend trades.

name = "12h_Donchian_Breakout_Trend_Volume"
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
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Donchian Channel (20) on 12h ===
    # We need 12h data to calculate Donchian, then align to 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels on 12h data
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned since we're using 12h data)
    # But we need to align to the original 12h price index
    donchian_high_12h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === Volume Spike (20-period) on 12h ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_12h[i]) or np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close breaks above upper Donchian + above 1d EMA200 + volume spike
            if close[i] > donchian_high_12h[i] and close[i] > ema_200_12h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below lower Donchian + below 1d EMA200 + volume spike
            elif close[i] < donchian_low_12h[i] and close[i] < ema_200_12h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below lower Donchian or trend change (below 1d EMA200)
            if close[i] < donchian_low_12h[i] or close[i] < ema_200_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper Donchian or trend change (above 1d EMA200)
            if close[i] > donchian_high_12h[i] or close[i] > ema_200_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals