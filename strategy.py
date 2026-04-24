#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Squeeze Breakout with 1d Volume Regime Filter.
- Bollinger Bands width percentile identifies low volatility squeezes
- Breakout occurs when price closes outside Bollinger Bands after squeeze
- 1d volume regime filter: only trade when 1d volume is above its 50-period median (high conviction)
- Works in both bull and bear markets as squeeze breakouts capture volatility expansion
- Uses 6h primary timeframe with 1d HTF for volume regime to avoid low-volume false breakouts
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = ma + (bb_std * bb_std_dev)
    lower_band = ma - (bb_std * bb_std_dev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Width Percentile (50-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB Width below 20th percentile (low volatility)
    squeeze = bb_width_percentile < 20
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # 1d Volume Regime Filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 1d volume median over 50 periods
    vol_median_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).median().values
    volume_regime = volume_1d > vol_median_1d  # High volume regime
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ma[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for squeeze breakout with volume confirmation
            # Need squeeze in previous bar and breakout in current bar
            if i > 0 and squeeze[i-1]:
                if breakout_up[i] and volume_regime_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif breakout_down[i] and volume_regime_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle band or squeeze breaks down
            if close[i] <= ma[i] or (i > 0 and not squeeze[i-1] and breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle band or squeeze breaks up
            if close[i] >= ma[i] or (i > 0 and not squeeze[i-1] and breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_1dVolumeRegime_v1"
timeframe = "6h"
leverage = 1.0