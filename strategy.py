#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Volume Regime Filter.
- Bollinger Bands (20,2) on 6h: squeeze when BB width < 20th percentile of last 50 bars
- Breakout long when close > upper band AND volume > 1.5 * 20-period average
- Breakout short when close < lower band AND volume > 1.5 * 20-period average
- 1d HTF volume regime: only trade when 1d volume > 20-period average (avoid low-volume chop)
- Exit when price returns to middle band (20-period SMA) or opposite band is touched
- Designed to catch volatility expansion after contraction, works in both bull and bear markets
- Volume confirmation avoids false breakouts; 1d volume regime filters ranging conditions
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on 6h
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_20 + (bb_std * std_20)
    lower_band = sma_20 - (bb_std * std_20)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Squeeze: width < 20th percentile of last 50 bars
    def rolling_percentile(arr, window, percentile):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                result[i] = np.percentile(valid_data, percentile)
        return result
    
    bb_width_20th = rolling_percentile(bb_width, 50, 20)
    squeeze = bb_width < bb_width_20th
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Calculate 1d volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime_1d = volume_1d > vol_ma_1d  # High volume regime
    volume_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20, 30) + 1  # Need BB, volume MA, and 1d data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(squeeze[i]) or np.isnan(volume_confirm[i]) or np.isnan(volume_regime_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: squeeze breakout above upper band AND volume confirmation AND 1d high volume regime
            if squeeze[i-1] and close[i] > upper_band[i] and volume_confirm[i] and volume_regime_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: squeeze breakout below lower band AND volume confirmation AND 1d high volume regime
            elif squeeze[i-1] and close[i] < lower_band[i] and volume_confirm[i] and volume_regime_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle band OR touches lower band (mean reversion)
            if close[i] <= sma_20[i] or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle band OR touches upper band (mean reversion)
            if close[i] >= sma_20[i] or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_1dVolumeRegime_v1"
timeframe = "6h"
leverage = 1.0