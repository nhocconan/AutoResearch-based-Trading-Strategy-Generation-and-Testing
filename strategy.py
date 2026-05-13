#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and 1w volume spike confirmation.
# Long when price breaks above Camarilla R3 AND price > 1d EMA50 AND 1w volume > 2.0 * 20-period average volume.
# Short when price breaks below Camarilla S3 AND price < 1d EMA50 AND 1w volume > 2.0 * 20-period average volume.
# Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium).
# Uses discrete position sizing (0.30) to balance return and risk. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 80-120 total trades over 4 years (20-30/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_1wVolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1w volume spike filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1w > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike.astype(float))
    
    # Calculate Camarilla pivot levels from 1d OHLC (HTF)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    #            Pivot = (high + low + close) / 3
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 1:
        return np.zeros(n)
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    camarilla_range = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + 1.1 * camarilla_range
    camarilla_s3 = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after first bar to have prior Camarilla levels
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND price > 1d EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Camarilla S3 AND price < 1d EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Camarilla Pivot (mean reversion)
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price returns to Camarilla Pivot (mean reversion)
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals