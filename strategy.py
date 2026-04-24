#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Regime Filter.
- Identifies low volatility periods (BB Width < 20th percentile) on 6h
- Breakout triggers when price closes outside BB(20,2) AND 6h volume > 2.0 * 20-period average
- 1d volume regime filter: only trade when 1d volume is above its 50-period median (high conviction)
- Long when close > upper band, Short when close < lower band
- Exit when price returns to middle band (mean reversion) or squeeze ends (BB Width > 50th percentile)
- Works in both bull and bear markets by trading volatility expansion after contraction
- Signal size: 0.25 discrete levels
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Bollinger Band Squeeze detection: low volatility regime
    # Squeeze when BB Width is below 20th percentile of last 100 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=100, min_periods=100).quantile(0.20).values
    squeeze_on = bb_width < bb_width_pct
    squeeze_off = bb_width > (bb_width_series.rolling(window=100, min_periods=100).quantile(0.50).values)
    
    # Volume confirmation on 6h: volume > 2.0 * 20-period average
    vol_ma = close_s.rolling(window=20, min_periods=20).mean().values  # Using close series for rolling
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # 1d volume regime filter: high conviction when 1d volume above 50-period median
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_median_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).median().values
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    volume_regime = volume_1d > vol_median_1d_aligned  # High volume conviction regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 100) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze_on[i]) or np.isnan(volume_confirm[i]) or np.isnan(volume_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: squeeze breaking upward with volume confirmation in high conviction regime
            if squeeze_on[i] and close[i] > upper_band[i] and volume_confirm[i] and volume_regime[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breaking downward with volume confirmation in high conviction regime
            elif squeeze_on[i] and close[i] < lower_band[i] and volume_confirm[i] and volume_regime[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle band OR squeeze ends (volatility expansion)
            if close[i] <= basis[i] or squeeze_off[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle band OR squeeze ends
            if close[i] >= basis[i] or squeeze_off[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_1dVolumeRegime_v1"
timeframe = "6h"
leverage = 1.0