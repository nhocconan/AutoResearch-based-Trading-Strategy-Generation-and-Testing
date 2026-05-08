#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA(13/34) crossover with 1d VWAP deviation and volume spike
# Uses EMA crossover for momentum, 1d VWAP deviation for mean reversion context,
# and volume spike for institutional confirmation. Works in both bull/bear by
# requiring volume confirmation and filtering extremes. Targets ~20-30 trades/year.

name = "6h_EMA13_34_1dVWAPDev_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 13 and 34 period EMA on 6h close
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).values
    
    # Calculate 1d VWAP
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    vwap_numerator = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_denominator = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(vwap_denominator, np.nan), 
                        where=vwap_denominator!=0)
    
    # Calculate deviation from VWAP (%)
    close_1d = df_1d['close'].values
    vwap_dev = (close_1d - vwap_1d) / vwap_1d * 100
    
    # Align EMA, VWAP deviation to 6h
    ema13_6h = align_htf_to_ltf(prices, close_series, ema13)
    ema34_6h = align_htf_to_ltf(prices, close_series, ema34)
    vwap_dev_6h = align_htf_to_ltf(prices, df_1d, vwap_dev)
    
    # Volume spike detection (24-period ~4d average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    vol_std = pd.Series(volume).rolling(window=24, min_periods=24).std()
    vol_spike = volume > (vol_ma.values + 2.0 * vol_std.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_6h[i]) or np.isnan(ema34_6h[i]) or 
            np.isnan(vwap_dev_6h[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: EMA13 > EMA34, price below VWAP (mean reversion), volume spike
            if ema13_6h[i] > ema34_6h[i] and vwap_dev_6h[i] < -0.5 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: EMA13 < EMA34, price above VWAP, volume spike
            elif ema13_6h[i] < ema34_6h[i] and vwap_dev_6h[i] > 0.5 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA crossover down or price reverts to VWAP
            if ema13_6h[i] < ema34_6h[i] or vwap_dev_6h[i] > 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA crossover up or price reverts to VWAP
            if ema13_6h[i] > ema34_6h[i] or vwap_dev_6h[i] < -0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals