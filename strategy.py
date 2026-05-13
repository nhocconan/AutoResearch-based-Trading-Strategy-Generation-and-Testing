#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike, and chop regime filter.
# Long when price breaks above Camarilla R3, close > 1d EMA34, volume > 2x average, and CHOP > 61.8 (range).
# Short when price breaks below Camarilla S3, close < 1d EMA34, volume > 2x average, and CHOP > 61.8.
# Uses discrete sizing 0.25 to target 75-200 total trades over 4 years on 4h timeframe.
# Camarilla levels provide intraday support/resistance; 1d EMA34 filters trend; volume confirms breakout strength;
# CHOP > 61.8 ensures range-bound conditions where mean reversion at extremes works best in bear markets.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Chop"
timeframe = "4h"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (based on prior day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # We need prior day's OHLC - get from 1d data
    df_1d_full = get_htf_data(prices, '1d')
    if len(df_1d_full) < 2:
        return np.zeros(n)
    # Shift 1d data by 1 to get prior completed day
    prior_close = df_1d_full['close'].shift(1).values
    prior_high = df_1d_full['high'].shift(1).values
    prior_low = df_1d_full['low'].shift(1).values
    # Align prior day's Camarilla levels to 4h timeframe
    camarilla_multiplier = 1.125  # for R3/S3
    prior_range = prior_high - prior_low
    camarilla_r3 = prior_close + camarilla_multiplier * prior_range
    camarilla_s3 = prior_close - camarilla_multiplier * prior_range
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_full, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_full, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback = 20
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Calculate Choppiness Index (CHOP) - range regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(lookback)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    min_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    max_min_range = max_high - min_low
    sum_atr = pd.Series(atr).rolling(window=lookback, min_periods=lookback).sum().values
    
    # Avoid division by zero and log of zero
    chop = np.full(n, np.nan)
    valid = (max_min_range > 0) & ~np.isnan(sum_atr) & ~np.isnan(max_min_range)
    chop[valid] = 100 * np.log10(sum_atr[valid] / max_min_range[valid]) / np.log10(lookback)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike, chop > 61.8 (range)
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i] and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike, chop > 61.8 (range)
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i] and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average OR chop < 38.2 (trend)
            if (low[i] < camarilla_s3_aligned[i] or 
                volume[i] < avg_volume[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average OR chop < 38.2 (trend)
            if (high[i] > camarilla_r3_aligned[i] or 
                volume[i] < avg_volume[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals