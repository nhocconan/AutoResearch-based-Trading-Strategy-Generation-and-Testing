#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# Uses 1d Donchian upper/lower channels derived from 20-period highs/lows.
# Enters long when price breaks above Donchian upper with volume and 1d ATR > 10th percentile.
# Enters short when price breaks below Donchian lower with volume and 1d ATR > 10th percentile.
# Designed to capture volatility expansion breakouts with low turnover (target: 12-37 trades/year).
# Works in bull markets (breakout momentum) and bear markets (volatility expansion during panic).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_high = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first period
    tr3[0] = tr1[0]  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 10th percentile of ATR for volatility filter (avoid low volatility periods)
    atr_10th = np.percentile(atr_1d[~np.isnan(atr_1d)], 10) if np.sum(~np.isnan(atr_1d)) > 0 else 0
    
    # Align 1d indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for Donchian(20) and ATR(14)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(atr_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Volatility filter: ATR above 10th percentile (avoid low volatility chop)
        vol_filter = atr_12h[i] > atr_10th
        
        # Price relative to Donchian levels
        price_above_upper = close[i] > donchian_high_12h[i]
        price_below_lower = close[i] < donchian_low_12h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume and volatility
            if (price_above_upper and volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume and volatility
            elif (price_below_lower and volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower
            if close[i] < donchian_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper
            if close[i] > donchian_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Volume"
timeframe = "12h"
leverage = 1.0