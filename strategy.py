#!/usr/bin/env python3
"""
4h_Keltner_Breakout_T2_WMA200_Trend_Volume
Hypothesis: Keltner Channel upper/lower bands (ATR-based) act as dynamic support/resistance. Breakouts above upper band or below lower band with volume confirmation and aligned 1d WMA200 trend signal continuation. Designed for 4h timeframe with low trade frequency (~20-40/year) to minimize fee drag. Works in both bull and bear markets by using trend-aligned breakouts and volatility-based channels that adapt to market conditions.
"""

name = "4h_Keltner_Breakout_T2_WMA200_Trend_Volume"
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
    
    # Get 1d data for WMA200 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Keltner Channel: ATR(10) * 2
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper = close + (2 * atr)
    keltner_lower = close - (2 * atr)
    
    # 1d trend filter: WMA(200) on close
    close_1d = df_1d['close'].values
    wma200_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 199:
            wma200_1d[i] = np.nan
        else:
            weights = np.arange(1, 201)
            wma200_1d[i] = np.dot(close_1d[i-199:i+1], weights) / weights.sum()
    wma200_1d = np.where(np.isnan(wma200_1d), close_1d, wma200_1d)  # fill NaN with close for early periods
    wma200_1d_aligned = align_htf_to_ltf(prices, df_1d, wma200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average (5 days in 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Keltner upper band, volume confirmation, price above 1d WMA200 (uptrend)
            if (close[i] > keltner_upper[i] and 
                volume_filter[i] and 
                close[i] > wma200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Keltner lower band, volume confirmation, price below 1d WMA200 (downtrend)
            elif (close[i] < keltner_lower[i] and 
                  volume_filter[i] and 
                  close[i] < wma200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Keltner upper band (failed breakout) OR volume drops
            if (close[i] < keltner_upper[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Keltner lower band (failed breakdown) OR volume drops
            if (close[i] > keltner_lower[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals