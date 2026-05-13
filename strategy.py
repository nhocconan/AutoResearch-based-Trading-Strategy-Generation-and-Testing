#!/usr/bin/env python3
# 4h_Price_Volume_Ratio_Spike_Trend_Filter
# Hypothesis: Volume spikes with price rejection from key levels (VWAP, SMA) signal exhaustion and reversals.
# Uses volume/price ratio > 2.0 as spike indicator, with 4h VWAP for context and 12h EMA for trend filter.
# Works in bull (pullbacks in uptrend) and bear (bounces in downtrend) by trading reversals.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Price_Volume_Ratio_Spike_Trend_Filter"
timeframe = "4h"
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
    
    # Calculate VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_cumsum = np.cumsum(vwap_numerator)
    volume_cumsum = np.cumsum(volume)
    vwap = vwap_cumsum / volume_cumsum
    
    # Price/volume ratio spike: volume > 2x average AND price deviation from VWAP
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    price_deviation = np.abs(close - vwap) / vwap  # Normalized deviation
    volume_spike = volume > (2.0 * vol_ma)
    price_rejection = price_deviation > 0.008  # 0.8% deviation from VWAP
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Price rejection below VWAP (long wick) with volume spike in uptrend
            if close[i] < vwap[i] and low[i] < vwap[i] * 0.992 and volume_spike[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejection above VWAP (short wick) with volume spike in downtrend
            elif close[i] > vwap[i] and high[i] > vwap[i] * 1.008 and volume_spike[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back above VWAP or trend weakens
            if close[i] > vwap[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back below VWAP or trend weakens
            if close[i] < vwap[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals