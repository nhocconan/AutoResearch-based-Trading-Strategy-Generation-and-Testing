#!/usr/bin/env python3
# 6h_1dTrend_VWAP_MeanReversion_WithVolFilter
# Hypothesis: Mean reversion to daily VWAP during 6h pullbacks in direction of 1d trend, confirmed by volume.
# Long when price pulls back to daily VWAP in uptrend with volume confirmation; short when price rallies to VWAP in downtrend.
# Uses daily VWAP as dynamic support/resistance. Trend filter ensures alignment with higher timeframe momentum.
# Volume filter prevents whipsaws in low-volume environments. Works in both bull and bear markets.

name = "6h_1dTrend_VWAP_MeanReversion_WithVolFilter"
timeframe = "6h"
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

    # Get daily data for VWAP and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily VWAP calculation (typical price * volume) cumulative
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_numerator = np.cumsum(typical_price * df_1d['volume'].values)
    vwap_denominator = np.cumsum(df_1d['volume'].values)
    # Avoid division by zero
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5 * 6-period average (1.5 days worth at 6h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_filter = volume > 1.5 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near VWAP (within 0.5%) + daily uptrend + volume filter
            if (abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] < 0.005 and 
                close[i] > ema50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near VWAP (within 0.5%) + daily downtrend + volume filter
            elif (abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] < 0.005 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves away from VWAP (>1%) OR trend reversal
            if (abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] > 0.01 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves away from VWAP (>1%) OR trend reversal
            if (abs(close[i] - vwap_aligned[i]) / vwap_aligned[i] > 0.01 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals