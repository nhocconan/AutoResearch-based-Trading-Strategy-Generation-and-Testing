#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversal_With_Volume_Filter
Hypothesis: Weekly pivot levels (calculated from prior week) act as strong support/resistance on 6h timeframe.
Price rejection at weekly R2/S2 with volume confirmation and aligned 1d trend (close > EMA34) signals reversal.
Works in bull markets via buying dips at support and in bear markets via selling rallies at resistance.
Targets 15-30 trades/year to minimize fee drag. Uses 0.25 position size.
"""

name = "6h_Weekly_Pivot_Reversal_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R2 = Pivot + (H - L)
    # S2 = Pivot - (H - L)
    prev_weekly_high = df_w['high'].shift(1).values
    prev_weekly_low = df_w['low'].shift(1).values
    prev_weekly_close = df_w['close'].shift(1).values
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_r2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)
    weekly_s2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r2_aligned = align_htf_to_ltf(prices, df_w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_w, weekly_s2)
    
    # 1d trend filter: EMA(34) on close
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after warmup
        if position == 0:
            # LONG: Rejection at S2 (price > S2 after touching or going below) with volume and uptrend
            if (close[i] > weekly_s2_aligned[i] and 
                low[i] <= weekly_s2_aligned[i] and  # touched or went below S2
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Rejection at R2 (price < R2 after touching or going above) with volume and downtrend
            elif (close[i] < weekly_r2_aligned[i] and 
                  high[i] >= weekly_r2_aligned[i] and  # touched or went above R2
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R2 or trend reverses
            if (close[i] >= weekly_r2_aligned[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S2 or trend reverses
            if (close[i] <= weekly_s2_aligned[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals