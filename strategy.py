#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Reversion
# Hypothesis: Mean-revert from weekly Camarilla pivot extremes (R4/S4) in the direction of daily trend.
# In uptrend (price > daily EMA50), buy at S4 (strong support) with confirmation from RSI < 30.
# In downtrend (price < daily EMA50), sell at R4 (strong resistance) with confirmation from RSI > 70.
# Weekly pivots provide strong institutional levels; daily trend filters ensure we trade with momentum.
# RSI prevents catching falling knives. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Low frequency due to requirement of hitting weekly extremes + trend alignment.

name = "6h_WeeklyPivot_DailyTrend_Reversion"
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

    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_weeks(prices, '1w')
    
    # Calculate weekly Camarilla levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # S4 = C - Range * 1.1/2
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    r4 = weekly_close + weekly_range * 1.1 / 2.0
    s4 = weekly_close - weekly_range * 1.1 / 2.0
    
    # Get daily data for trend and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align weekly levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Align daily indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S4 + daily uptrend + RSI oversold
            if close[i] <= s4_aligned[i] and close[i] > ema50_1d_aligned[i] and rsi_aligned[i] < 30:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R4 + daily downtrend + RSI overbought
            elif close[i] >= r4_aligned[i] and close[i] < ema50_1d_aligned[i] and rsi_aligned[i] > 70:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot OR RSI overbought
            if close[i] >= weekly_pivot_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot OR RSI oversold
            if close[i] <= weekly_pivot_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

# Helper function to get weekly data (since not in standard timeframes)
def get_htf_weeks(prices, timeframe):
    from mtf_data import get_htf_data
    return get_htf_data(prices, timeframe)