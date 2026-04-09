#!/usr/bin/env python3
# mtf_1h_ema_cross_volume_regime_v1
# Hypothesis: 1h EMA(9)/EMA(21) cross with 4h EMA(50) trend filter, 1d ATR regime filter, and volume spike confirmation.
# Enters long when 1h EMA9 crosses above EMA21, price > 4h EMA50, 1d ATR > 20-period median ATR (high volatility regime), and volume > 1.5x 20-period average.
# Enters short when 1h EMA9 crosses below EMA21, price < 4h EMA50, same regime and volume filters.
# Uses discrete position sizing (±0.20) to minimize fee churn. Session filter 08-20 UTC to reduce noise.
# Target: 60-150 total trades over 4 years (15-37/year). Works in bull/bear via trend + regime filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_cross_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h EMA9 and EMA21
    ema9 = pd.Series(close).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Get 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema4h_50 = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema4h_50)
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_median_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).median().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # Volume spike detection (20-period volume average on 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or
            np.isnan(ema4h_50_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_median_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in high volatility (ATR > median ATR)
        high_vol_regime = atr_1d_aligned[i] > atr_median_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA9 crosses below EMA21 or price < 4h EMA50
            if ema9[i] < ema21[i] or close[i] < ema4h_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA9 crosses above EMA21 or price > 4h EMA50
            if ema9[i] > ema21[i] or close[i] > ema4h_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: EMA9 crosses above EMA21, price > 4h EMA50, high vol regime, volume spike
            if (ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and  # bullish cross
                close[i] > ema4h_50_aligned[i] and
                high_vol_regime and
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Enter short: EMA9 crosses below EMA21, price < 4h EMA50, high vol regime, volume spike
            elif (ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and  # bearish cross
                  close[i] < ema4h_50_aligned[i] and
                  high_vol_regime and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals

def calculate_atr(high, low, close, window):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: use only high-low
    atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    return atr