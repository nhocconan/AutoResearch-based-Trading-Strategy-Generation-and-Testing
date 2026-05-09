#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day ATR-based volatility regime and 1-week volume-weighted average price (VWAP) trend filter.
# Uses 1-day ATR(14) normalized by 50-period mean to detect low-volatility regimes (mean-reversion favorable).
# Enters long when price crosses above VWAP(5) in low-volatility regime, short when below.
# Exits when volatility regime shifts to high volatility or price reverts to VWAP.
# Weekly VWAP provides stable trend reference that adapts to both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_ATR_VolRegime_VWAP5_Trend"
timeframe = "12h"
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
    
    # Calculate 1-day ATR(14) for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range components
    prev_close = np.roll(df_1d['close'], 1)
    prev_close[0] = np.nan
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - prev_close)
    tr3 = np.abs(df_1d['low'] - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 50-period mean of ATR for normalization
    atr_mean_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: low volatility when ATR < 0.8 * mean ATR
    vol_regime_low = atr_14 < (0.8 * atr_mean_50)
    vol_regime_low_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_low)
    
    # Calculate 1-week VWAP (5-period VWAP on 6d data approximates 1-week)
    # For 12h timeframe, 1 week = 14 bars (7 days * 2 bars/day)
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=14, min_periods=14).sum().values
    vwap_den = pd.Series(volume).rolling(window=14, min_periods=14).sum().values
    vwap = vwap_num / vwap_den
    
    # Price position relative to VWAP
    price_above_vwap = close > vwap
    price_below_vwap = close < vwap
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_low_aligned[i]) or
            np.isnan(price_above_vwap[i]) or np.isnan(price_below_vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: low volatility regime + price above VWAP
            if vol_regime_low_aligned[i] and price_above_vwap[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: low volatility regime + price below VWAP
            elif vol_regime_low_aligned[i] and price_below_vwap[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime shifts to high OR price crosses below VWAP
            if (not vol_regime_low_aligned[i]) or (not price_above_vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime shifts to high OR price crosses above VWAP
            if (not vol_regime_low_aligned[i]) or (not price_below_vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals