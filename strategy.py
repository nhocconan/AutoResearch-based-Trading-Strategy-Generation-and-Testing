# 1D_WMMA_TREND_REVERSAL
# WMMA-weighted mean deviation mean reversion on daily timeframe with weekly trend filter
# Long when price deviates significantly below WMMA in weekly uptrend
# Short when price deviates significantly above WMMA in weekly downtrend
# Uses deviation bands (2.5 * ATR) for entry and mean reversion to WMMA for exit
# Designed for low trade frequency (<25/year) to minimize fee drag
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate WMMA on weekly close (Weighted Moving Average)
    close_1w = df_1w['close'].values
    length = 9
    weights = np.arange(1, length + 1)
    wma_1w = np.full_like(close_1w, np.nan)
    
    for i in range(length - 1, len(close_1w)):
        wma_1w[i] = np.dot(close_1w[i - length + 1:i + 1], weights) / weights.sum()
    
    # Weekly trend: slope of WMMA
    wma_slope = np.diff(wma_1w, prepend=np.nan)
    wma_trend_up = wma_slope > 0
    wma_trend_down = wma_slope < 0
    
    # Align weekly trend to daily
    wma_trend_up_aligned = align_htf_to_ltf(prices, df_1w, wma_trend_up.astype(float))
    wma_trend_down_aligned = align_htf_to_ltf(prices, df_1w, wma_trend_down.astype(float))
    
    # Calculate WMMA on daily for entry/exit
    wma_daily = np.full_like(close, np.nan)
    for i in range(length - 1, len(close)):
        wma_daily[i] = np.dot(close[i - length + 1:i + 1], weights) / weights.sum()
    
    # ATR for deviation bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan)
    
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Deviation from WMMA
    deviation = close - wma_daily
    upper_band = wma_daily + 2.5 * atr
    lower_band = wma_daily - 2.5 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(wma_trend_up_aligned[i]) or np.isnan(wma_trend_down_aligned[i]) or
            np.isnan(wma_daily[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below lower band in weekly uptrend
            if close[i] < lower_band[i] and wma_trend_up_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price above upper band in weekly downtrend
            elif close[i] > upper_band[i] and wma_trend_down_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to WMMA (mean reversion)
            if position == 1:
                # Exit long: Price crosses above WMMA
                if close[i] > wma_daily[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price crosses below WMMA
                if close[i] < wma_daily[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WMMA_Trend_Reversal"
timeframe = "1d"
leverage = 1.0