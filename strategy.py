#!/usr/bin/env python3
# 6h_VWAP_Deviation_12hTrend_Volume
# Hypothesis: Price reverts to VWAP in ranging markets but breaks out with trend.
# Long when price crosses above VWAP with 12h uptrend and volume spike.
# Short when price crosses below VWAP with 12h downtrend and volume spike.
# VWAP acts as dynamic support/resistance. Trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation, reducing false signals.
# Works in bull markets (breakouts above VWAP in uptrend) and bear markets (breakdowns below VWAP in downtrend).
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "6h_VWAP_Deviation_12hTrend_Volume"
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

    # Calculate VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = cum_tpv / cum_vol
    # Handle division by zero at start
    vwap = np.where(cum_vol == 0, typical_price, vwap)

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h trend: EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 6h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(vwap[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above VWAP + 12h uptrend + volume spike
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below VWAP + 12h downtrend + volume spike
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below VWAP or trend reversal
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above VWAP or trend reversal
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals