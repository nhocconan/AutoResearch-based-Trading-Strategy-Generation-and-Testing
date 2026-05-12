#!/usr/bin/env python3
"""
6h_1d_VWAP_Deviation_Reversion
Hypothesis: Mean reversion from daily VWAP deviations on 6h timeframe with trend filter.
In ranging markets (common in bear/consolidation), price tends to revert to the day's VWAP.
In trending markets, we filter trades to align with the 1d trend to avoid counter-trend whipsaws.
This strategy works in both bull and bear markets by combining mean reversion with trend alignment.
"""

name = "6h_1d_VWAP_Deviation_Reversion"
timeframe = "6h"
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
    
    # VWAP deviation: (close - VWAP) / VWAP
    # Calculate VWAP for each 6h bar using intraday data approximation
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    vwap_deviation = (close - vwap) / vwap
    
    # 1d data for trend filter and VWAP reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily VWAP from 1d data (approximated from OHLC)
    # Using typical price approximation for daily VWAP
    df_1d_typical = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Approximate daily VWAP using volume-weighted typical price
    daily_vwap_num = np.cumsum(df_1d_typical * df_1d['volume'])
    daily_vwap_den = np.cumsum(df_1d['volume'])
    daily_vwap = daily_vwap_num / daily_vwap_den
    daily_vwap_aligned = align_htf_to_ltf(prices, df_1d, daily_vwap.values)
    
    # Deviation from daily VWAP
    daily_vwap_deviation = (close - daily_vwap_aligned) / daily_vwap_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(daily_vwap_aligned[i]) or
            np.isnan(vwap_deviation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price below daily VWAP (oversold) AND price above 1d EMA50 (uptrend)
            if (daily_vwap_deviation[i] < -0.015 and  # 1.5% below daily VWAP
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above daily VWAP (overbought) AND price below 1d EMA50 (downtrend)
            elif (daily_vwap_deviation[i] > 0.015 and   # 1.5% above daily VWAP
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above daily VWAP OR closes below 1d EMA50
            if (daily_vwap_deviation[i] > 0.005 or   # Back to VWAP (0.5% above)
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below daily VWAP OR closes above 1d EMA50
            if (daily_vwap_deviation[i] < -0.005 or  # Back to VWAP (0.5% below)
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals