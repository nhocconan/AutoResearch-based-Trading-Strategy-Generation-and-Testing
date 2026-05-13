#!/usr/bin/env python3
# 4h_Price_Channel_Range_Momentum
# Hypothesis: Use 1d Donchian channel (55) for trend direction and 4h price action for mean reversion within the channel.
# Go long when price touches 4h lower Bollinger Band (20,2) in the upper half of 1d Donchian channel.
# Go short when price touches 4h upper Bollinger Band in the lower half of 1d Donchian channel.
# Requires volume > 1.5x 20-period average and ADX(14) < 25 (range-bound market).
# Exit when price reaches 4h middle Bollinger Band or Donchian midpoint.
# Works in bull/bear: captures mean reversion in ranging markets while avoiding strong trends.
# Low frequency due to range requirement and strict volume filter.

name = "4h_Price_Channel_Range_Momentum"
timeframe = "4h"
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

    # Get daily data for trend context
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian Channel (55-period) for trend context
    donch_high_1d = pd.Series(df_1d['high']).rolling(window=55, min_periods=55).max().values
    donch_low_1d = pd.Series(df_1d['low']).rolling(window=55, min_periods=55).min().values
    donch_mid_1d = (donch_high_1d + donch_low_1d) / 2
    
    # Align 1d Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # 4h Bollinger Bands (20, 2) for entry signals
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    middle_bb = sma20
    
    # 4h ADX (14) for regime filter - range when ADX < 25
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14 * 100 + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14 * 100 + 1e-10)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    # Pad arrays to match length
    plus_di_14 = np.concatenate([np.zeros(14), plus_di_14])
    minus_di_14 = np.concatenate([np.zeros(14), minus_di_14])
    adx_14 = np.concatenate([np.zeros(14), adx_14])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(55, n):  # Wait for Donchian to be valid
        # Skip if any required value is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(middle_bb[i]) or 
            np.isnan(adx_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches lower BB in upper half of 1d Donchian + range + volume
            if (close[i] <= lower_bb[i] and 
                close[i] > donch_mid_aligned[i] and 
                adx_14[i] < 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches upper BB in lower half of 1d Donchian + range + volume
            elif (close[i] >= upper_bb[i] and 
                  close[i] < donch_mid_aligned[i] and 
                  adx_14[i] < 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches middle BB or Donchian midpoint
            if close[i] >= middle_bb[i] or close[i] >= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches middle BB or Donchian midpoint
            if close[i] <= middle_bb[i] or close[i] <= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals