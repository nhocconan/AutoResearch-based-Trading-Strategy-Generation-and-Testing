#!/usr/bin/env python3
# 1d_Keltner_Channel_Breakout_Volume_Trend
# Hypothesis: Daily Keltner Channel breakouts with volume confirmation and weekly EMA trend filter capture momentum.
# Works in bull markets via breakouts with trend and in bear via filtered breakdowns against trend.
# Targets 15-25 trades/year to minimize fee drag.

name = "1d_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "1d"
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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate ATR for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate EMA for Keltner Channel middle line
    ema = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    keltner_upper = ema + 2 * atr
    keltner_lower = ema - 2 * atr
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above upper Keltner with volume confirmation and weekly EMA uptrend
            if close[i] > keltner_upper[i] and volume_filter[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower Keltner with volume confirmation and weekly EMA downtrend
            elif close[i] < keltner_lower[i] and volume_filter[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to EMA (middle) or weekly trend reverses
            if close[i] < ema[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to EMA (middle) or weekly trend reverses
            if close[i] > ema[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals