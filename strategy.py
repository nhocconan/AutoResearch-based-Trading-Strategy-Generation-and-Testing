#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Volume Regime Filter
- Bollinger Bands (20, 2.0) on 6h: squeeze when bandwidth < 20th percentile of last 50 bars
- Breakout long when price closes above upper band during squeeze release + 12h volume > 1.5x 20-period average
- Breakout short when price closes below lower band during squeeze release + 12h volume > 1.5x 20-period average
- Uses 6h timeframe targeting 12-30 trades/year (50-120 over 4 years)
- Works in bull markets via breakout continuation, in bear markets via breakdown continuations
- Volume regime filter prevents false breakouts in low-volume environments
"""

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
    
    # Get 12h data for volume regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h average volume for regime filter
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Bollinger Bands on 6h (20, 2.0)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Bollinger Band Width (normalized)
    bb_width = (upper_band - lower_band) / basis
    # Squeeze condition: bandwidth < 20th percentile of last 50 bars
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Squeeze release: previous bar was in squeeze, current bar is not
    squeeze_release = np.zeros(n, dtype=bool)
    squeeze_release[1:] = (squeeze_condition[:-1] == True) & (squeeze_condition[1:] == False)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # BB needs 20, percentile needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume regime filter: 12h volume > 1.5x 20-period average
        volume_regime = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Breakout conditions during squeeze release
        long_breakout = squeeze_release[i] and (close[i] > upper_band[i]) and volume_regime
        short_breakout = squeeze_release[i] and (close[i] < lower_band[i]) and volume_regime
        
        if position == 0:
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: mean reversion to middle band or opposite squeeze breakout
            exit_signal = False
            
            if position == 1:
                # Exit long: price closes below middle band (mean reversion)
                if close[i] < basis[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price closes above middle band (mean reversion)
                if close[i] > basis[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_12hVolumeRegime"
timeframe = "6h"
leverage = 1.0