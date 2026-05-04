#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ATR regime filter and volume confirmation
# Uses Donchian(20) from 12h chart for breakout signals, filtered by 1d ATR-based regime
# (low volatility = mean reversion at bands, high volatility = breakout continuation)
# and volume spike confirmation. Designed for 12-37 trades/year (~50-150 total over 4 years)
# to minimize fee drag. Works in bull markets via upward breakouts and in bear markets
# via downward breakouts, with regime filter preventing false signals in chop.

name = "12h_Donchian20_VolumeSpike_1dATR_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian(20) channels on 12h data
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for completed 12h bar)
    highest_20_aligned = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_20)
    
    # Get 1d data for ATR regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d data
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR percentile rank (20-period) for regime detection
    atr_percentile = pd.Series(atr_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Regime: ATR percentile > 0.5 = high volatility (breakout mode), < 0.5 = low volatility (mean reversion)
    high_vol_regime = atr_percentile_aligned > 0.5
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(high_vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In high volatility regime: breakout continuation
            # In low volatility regime: mean reversion at bands
            if high_vol_regime[i]:
                # High vol: breakout continuation
                if (close[i] > highest_20_aligned[i] and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < lowest_20_aligned[i] and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
            else:
                # Low vol: mean reversion at bands
                if (close[i] < lowest_20_aligned[i] and volume_spike[i]):
                    signals[i] = 0.25  # long at lower band
                    position = 1
                elif (close[i] > highest_20_aligned[i] and volume_spike[i]):
                    signals[i] = -0.25  # short at upper band
                    position = -1
        elif position == 1:
            # Exit long: price reaches opposite band or regime changes against position
            if (close[i] >= highest_20_aligned[i] or 
                (not high_vol_regime[i] and close[i] > lowest_20_aligned[i]) or
                (high_vol_regime[i] and close[i] < lowest_20_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches opposite band or regime changes against position
            if (close[i] <= lowest_20_aligned[i] or 
                (not high_vol_regime[i] and close[i] < highest_20_aligned[i]) or
                (high_vol_regime[i] and close[i] > highest_20_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals