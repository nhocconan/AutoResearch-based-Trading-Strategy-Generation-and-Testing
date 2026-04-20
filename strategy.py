#!/usr/bin/env python3
"""
1d_RangeBound_MeanReversion_With_Weekly_Filter
Hypothesis: In range-bound markets (weekly ATR < 30d ATR), price reverts to weekly VWAP.
Long when price < weekly VWAP - 0.5*weekly ATR, short when price > weekly VWAP + 0.5*weekly ATR.
Exit when price crosses weekly VWAP. Works in both bull/bear by adapting to weekly volatility regime.
Target: 50-100 total trades over 4 years (12-25/year) with position size 0.25.
"""

name = "1d_RangeBound_MeanReversion_With_Weekly_Filter"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    tp_weekly = (df_weekly['high'].values + df_weekly['low'].values + df_weekly['close'].values) / 3.0
    vol_weekly = df_weekly['volume'].values
    vwap_weekly = np.full_like(tp_weekly, np.nan)
    if len(tp_weekly) >= 1:
        cum_vol = 0
        cum_tpv = 0
        for i in range(len(tp_weekly)):
            cum_vol += vol_weekly[i]
            cum_tpv += tp_weekly[i] * vol_weekly[i]
            if cum_vol > 0:
                vwap_weekly[i] = cum_tpv / cum_vol
    vwap_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vwap_weekly)
    
    # Calculate weekly ATR (14-period)
    tr_weekly = np.maximum(
        df_weekly['high'].values - df_weekly['low'].values,
        np.maximum(
            np.abs(df_weekly['high'].values - np.roll(df_weekly['close'].values, 1)),
            np.abs(df_weekly['low'].values - np.roll(df_weekly['close'].values, 1))
        )
    )
    tr_weekly[0] = df_weekly['high'].values[0] - df_weekly['low'].values[0]
    atr_weekly = np.full_like(tr_weekly, np.nan)
    if len(tr_weekly) >= 14:
        atr_weekly[13] = np.mean(tr_weekly[:14])
        for i in range(14, len(tr_weekly)):
            atr_weekly[i] = (atr_weekly[i-1] * 13 + tr_weekly[i]) / 14
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    
    # Calculate daily ATR (30-period) for regime filter
    tr_daily = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr_daily[0] = high[0] - low[0]
    atr_daily = np.full_like(tr_daily, np.nan)
    if len(tr_daily) >= 30:
        atr_daily[29] = np.mean(tr_daily[:30])
        for i in range(30, len(tr_daily)):
            atr_daily[i] = (atr_daily[i-1] * 29 + tr_daily[i]) / 30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure daily ATR is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_weekly_aligned[i]) or np.isnan(atr_weekly_aligned[i]) or 
            np.isnan(atr_daily[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Range-bound regime: weekly ATR < 60% of daily ATR (low volatility)
        if atr_weekly_aligned[i] < 0.6 * atr_daily[i]:
            # Entry conditions
            if position == 0:
                # Long: price significantly below VWAP
                if close[i] < vwap_weekly_aligned[i] - 0.5 * atr_weekly_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price significantly above VWAP
                elif close[i] > vwap_weekly_aligned[i] + 0.5 * atr_weekly_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Long exit: price crosses above VWAP
                if close[i] > vwap_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Short exit: price crosses below VWAP
                if close[i] < vwap_weekly_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Trending regime: stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals