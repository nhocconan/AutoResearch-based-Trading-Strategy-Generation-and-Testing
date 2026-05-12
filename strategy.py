#!/usr/bin/env python3
# 4h_1D_Choppiness_Reversal
# Hypothesis: Mean reversion in ranging markets using daily Choppiness Index on 4H timeframe. 
# When market is choppy (CHOP > 61.8), price tends to revert to mean (daily VWAP).
# In trending markets (CHOP < 38.2), avoid trades to prevent whipsaw.
# Works in both bull and bear markets as ranging regimes occur in all market conditions.
# Uses volume confirmation to filter low-quality signals.
# Target: 20-40 trades per year by requiring chop regime + VWAP deviation + volume spike.

name = "4h_1D_Choppiness_Reversal"
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
    
    # Volume spike filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Choppiness Index and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily Choppiness Index (14-period)
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            )
        # Smooth TR using Wilder's smoothing (same as ATR)
        atr[period-1] = np.nanmean(tr[1:period+1])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # True range sum over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Max high and min low over period
        max_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        
        # Choppiness formula: 100 * log10(tr_sum / (max_high - min_low)) / log10(period)
        range_val = max_high - min_low
        chop = np.full_like(close_arr, np.nan)
        valid = (range_val > 0) & (~np.isnan(tr_sum))
        chop[valid] = 100 * np.log10(tr_sum[valid] / range_val[valid]) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Daily VWAP (typical price * volume / cumulative volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    vwap_align = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(chop_align[i]) or 
            np.isnan(vwap_align[i]) or 
            np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Choppy market + price below VWAP + volume spike
            if (chop_align[i] > 61.8 and 
                close[i] < vwap_align[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Choppy market + price above VWAP + volume spike
            elif (chop_align[i] > 61.8 and 
                  close[i] > vwap_align[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above VWAP OR chop drops below 38.2 (trending)
            if close[i] > vwap_align[i] or chop_align[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below VWAP OR chop drops below 38.2 (trending)
            if close[i] < vwap_align[i] or chop_align[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals