#!/usr/bin/env python3
# 1d_1W_Donchian20_Breakout_TrendVol
# Hypothesis: Daily breakouts from weekly Donchian(20) channels with weekly trend filter and volume confirmation.
# Long when price breaks above weekly upper band with volume spike and weekly uptrend (close > weekly SMA50).
# Short when price breaks below weekly lower band with volume spike and weekly downtrend (close < weekly SMA50).
# Uses tight entry conditions (trend + volume + breakout) to target 10-25 trades per year per symbol, avoiding overtrading.
# Works in bull markets via trend-following breakouts and in bear markets via mean-reversion at extreme weekly levels.

name = "1d_1W_Donchian20_Breakout_TrendVol"
timeframe = "1d"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly SMA50 for trend filter
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Weekly Donchian(20) from previous week
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    upper_band = pd.Series(prev_high_1w).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(prev_low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band + volume spike + close above weekly SMA50 (uptrend)
            if (close[i] > upper_band_aligned[i] and 
                volume_spike[i] and 
                close[i] > sma_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band + volume spike + close below weekly SMA50 (downtrend)
            elif (close[i] < lower_band_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < sma_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous week's H-L range OR closes below weekly SMA50
            if (close[i] < upper_band_aligned[i] and close[i] > lower_band_aligned[i]) or \
               close[i] < sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous week's H-L range OR closes above weekly SMA50
            if (close[i] < upper_band_aligned[i] and close[i] > lower_band_aligned[i]) or \
               close[i] > sma_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals