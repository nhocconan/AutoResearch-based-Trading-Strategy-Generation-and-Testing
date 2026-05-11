#!/usr/bin/env python3
name = "6h_ADX_Trend_Stochastic_MeanReversion_1dVol"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    plus_dm = np.diff(high)
    minus_dm = np.diff(low)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    plus_di = 100 * np.convolve(plus_dm, np.ones(period)/period, mode='same') / atr
    minus_di = 100 * np.convolve(minus_dm, np.ones(period)/period, mode='same') / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = np.zeros_like(dx)
    adx[period-1] = np.mean(dx[:period])
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = np.minimum.accumulate(low)
    highest_high = np.maximum.accumulate(high)
    for i in range(len(low)):
        if i < k_period:
            lowest_low[i] = np.min(low[:i+1])
            highest_high[i] = np.max(high[:i+1])
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = np.where((highest_high - lowest_low) == 0, 50, k)
    d = np.convolve(k, np.ones(d_period)/d_period, mode='same')
    return k, d

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend strength (strong trend >25)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    strong_trend = adx_1d_aligned > 25
    
    # 6h Stochastic for mean reversion signals
    k, d = calculate_stochastic(high, low, close, 14, 3)
    k_slow = np.convolve(k, np.ones(3)/3, mode='same')
    d_slow = np.convolve(d, np.ones(3)/3, mode='same')
    
    # 1d volume filter: volume > 1.3x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = np.convolve(vol_1d, np.ones(20)/20, mode='same')
    vol_ma20_1d[:19] = np.nan
    for i in range(19, len(vol_1d)):
        vol_ma20_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.3 * vol_ma20_1d_aligned
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(k_slow[i]) or np.isnan(d_slow[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # In strong trend: follow stochastic crossovers in trend direction
            if strong_trend[i]:
                # Uptrend: buy when stochastic crosses up from oversold
                if k_slow[i] > d_slow[i] and k_slow[i-1] <= d_slow[i-1] and k_slow[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Downtrend: sell when stochastic crosses down from overbought
                elif k_slow[i] < d_slow[i] and k_slow[i-1] >= d_slow[i-1] and k_slow[i] > 70:
                    signals[i] = -0.25
                    position = -1
            else:
                # In weak trend/ranging: mean reversion at extremes
                if k_slow[i] < 20 and k_slow[i] > d_slow[i] and k_slow[i-1] <= d_slow[i-1]:
                    signals[i] = 0.25
                    position = 1
                elif k_slow[i] > 80 and k_slow[i] < d_slow[i] and k_slow[i-1] >= d_slow[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: stochastic overbought or trend weakness
            if k_slow[i] > 80 or (strong_trend[i] and k_slow[i] < d_slow[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: stochastic oversold or trend weakness
            if k_slow[i] < 20 or (strong_trend[i] and k_slow[i] > d_slow[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals