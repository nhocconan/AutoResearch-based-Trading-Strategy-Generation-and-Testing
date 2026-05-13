#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1_S1_Breakout_WeeklyTrend
# Hypothesis: Price breaks above/below weekly Camarilla R1/S1 levels with
# weekly EMA34 trend confirmation and volume spike. Works in bull (buy R1 breaks in uptrend) 
# and bear (sell S1 breaks in downtrend). Low frequency due to weekly timeframe and
# strict volume confirmation, targeting 30-100 trades over 4 years.

name = "1d_Weekly_Camarilla_R1_S1_Breakout_WeeklyTrend"
timeframe = "1d"
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

    # Get weekly data for Camarilla pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Pivot point
    pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Range
    range_ = df_1w['high'] - df_1w['low']
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Weekly trend filter: EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all weekly indicators to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Price levels
        price = close[i]
        
        # Trend conditions
        uptrend = price > ema34_1w_aligned[i]
        downtrend = price < ema34_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price breaks above R1 + uptrend + volume spike
            if price > r1_aligned[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + downtrend + volume spike
            elif price < s1_aligned[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 OR trend reversal
            if price < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 OR trend reversal
            if price > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals