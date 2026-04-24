#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d Williams Alligator trend filter and volume confirmation.
- Primary timeframe: 6h for lower trade frequency and better signal quality vs lower TFs.
- HTF: 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) for trend direction.
  Trend is bullish when Lips > Teeth > Jaw, bearish when Jaw > Teeth > Lips.
- Volume: Current 6h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Donchian: Upper/lower bands from 20-period high/low.
- Entry: Long when price breaks above Upper band AND Alligator bullish AND volume spike.
         Short when price breaks below Lower band AND Alligator bearish AND volume spike.
- Exit: Price reverts to mid-band (average of upper/lower) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy combines institutional volume confirmation with Donchian breakouts,
filtered by daily Alligator trend to avoid counter-trend trades. Works in both bull and bear markets
by only taking trades in the direction of the 1d trend, with volume spikes confirming
participation. Donchian levels provide clear structure for entries and exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least Jaw period
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    df_1d_close = df_1d['close'].values
    jaw = pd.Series(df_1d_close).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(df_1d_close).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(df_1d_close).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Alligator trend: bullish when Lips > Teeth > Jaw, bearish when Jaw > Teeth > Lips
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (jaw > teeth) & (teeth > lips)
    
    # Calculate 20-period 1d volume MA for volume confirmation
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian bands (20-period) on 6h data
    # Upper band = 20-period high, Lower band = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2  # Mid-band for exit
    
    # Align HTF indicators to 6h
    alligator_bullish_aligned = align_htf_to_ltf(prices, df_1d, alligator_bullish.astype(float))
    alligator_bearish_aligned = align_htf_to_ltf(prices, df_1d, alligator_bearish.astype(float))
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 20)  # Donchian20, Alligator Jaw13, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(alligator_bullish_aligned[i]) or np.isnan(alligator_bearish_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > Upper band AND Alligator bullish
                if curr_close > donchian_upper[i] and alligator_bullish_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price < Lower band AND Alligator bearish
                elif curr_close < donchian_lower[i] and alligator_bearish_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to mid-band OR loss of volume confirmation
            if curr_close <= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to mid-band OR loss of volume confirmation
            if curr_close >= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dAlligator_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0