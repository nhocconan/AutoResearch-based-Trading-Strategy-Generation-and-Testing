#!/usr/bin/env python3
name = "1D_Keltner_MR_With_WaveTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D INDICATORS ===
    # EMA for Keltner
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR for Keltner width
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper = ema20 + 1.5 * atr
    lower = ema20 - 1.5 * atr
    
    # WaveTrend (WT) on 1d
    hlc3 = (high + low + close) / 3.0
    esa = pd.Series(hlc3).ewm(span=10, adjust=False, min_periods=10).mean().values
    d = np.abs(hlc3 - esa)
    de = pd.Series(d).ewm(span=10, adjust=False, min_periods=10).mean().values
    de = np.where(de == 0, 0.001, de)
    ci = (hlc3 - esa) / (0.015 * de)
    wt1 = pd.Series(ci).ewm(span=21, adjust=False, min_periods=21).mean().values
    wt2 = pd.Series(wt1).ewm(span=4, adjust=False, min_periods=4).mean().values
    wt_signal = wt1 - wt2
    
    # === 1W TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === VOLUME SPIKE FILTER ===
    vol_avg = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(wt_signal[i]) or np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Mean reversion long: price below lower Keltner + WT oversold + weekly uptrend + volume spike
            if (close[i] < lower[i] and
                wt_signal[i] < -50 and
                close[i] > ema34_1w_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Mean reversion short: price above upper Keltner + WT overbought + weekly downtrend + volume spike
            elif (close[i] > upper[i] and
                  wt_signal[i] > 50 and
                  close[i] < ema34_1w_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above EMA20 or WT crosses above zero
            if close[i] > ema20[i] or wt_signal[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below EMA20 or WT crosses below zero
            if close[i] < ema20[i] or wt_signal[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals