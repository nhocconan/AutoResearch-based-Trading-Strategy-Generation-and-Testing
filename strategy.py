#!/usr/bin/env python3
"""
Hypothesis: 6h market regime (trend vs range) detected by R-squared of linear regression
on price over 20 periods. In trending regimes (R² > 0.6), trade 60-period Donchian
breakouts with volume confirmation. In ranging regimes (R² < 0.4), fade at 2-standard
deviation Bollinger Bands with mean reversion. Uses volume filter to avoid false
signals. Designed to work in both bull and bear markets by adapting to regime.
Target: 60-120 trades over 4 years (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _rsquared(x, y):
    """Compute R-squared of linear regression y = a*x + b."""
    if len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return 0.0
    a = num / den
    b = y_mean - a * x_mean
    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 60-period linear regression R-squared for regime detection
    lookback_regime = 60
    r_squared = np.full(n, np.nan)
    for i in range(lookback_regime, n):
        x = np.arange(lookback_regime)
        y = close[i-lookback_regime:i]
        r_squared[i] = _rsquared(x, y)
    
    # 60-period Donchian channels for breakouts
    lookback_donch = 60
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback_donch, n):
        highest_high[i] = np.max(high[i-lookback_donch:i])
        lowest_low[i] = np.min(low[i-lookback_donch:i])
    
    # 20-period Bollinger Bands for mean reversion
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std_dev[i] = np.std(close[i-bb_period:i])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
    
    # 20-period average volume for spike/filters
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need enough data for all indicators
    start_idx = max(lookback_regime, lookback_donch, bb_period, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(r_squared[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        vol_filter = vol_ratio > 1.5  # Require above-average volume
        
        if position == 0:
            # Regime: trending (R² > 0.6) -> breakout
            if r_squared[i] > 0.6 and vol_filter:
                if price > highest_high[i]:
                    signals[i] = size
                    position = 1
                elif price < lowest_low[i]:
                    signals[i] = -size
                    position = -1
            # Regime: ranging (R² < 0.4) -> mean reversion at Bollinger Bands
            elif r_squared[i] < 0.4 and vol_filter:
                if price >= upper_band[i]:
                    signals[i] = -size
                    position = -1
                elif price <= lower_band[i]:
                    signals[i] = size
                    position = 1
        elif position == 1:
            # Long exit: reverse signal or volatility expansion
            if (r_squared[i] < 0.4 and price <= sma[i]) or \
               (r_squared[i] > 0.6 and price < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: reverse signal or volatility expansion
            if (r_squared[i] < 0.4 and price >= sma[i]) or \
               (r_squared[i] > 0.6 and price > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6r_RegimeAdaptive_DonchianBB_VolumeFilter"
timeframe = "6h"
leverage = 1.0