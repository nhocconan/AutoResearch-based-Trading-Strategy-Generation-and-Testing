#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index regime filter + 1d Donchian breakout + volume confirmation
# Choppiness Index identifies trending vs ranging markets. In trending regimes (CHOP < 38.2),
# we take Donchian breakouts in the direction of the 1d trend. In ranging regimes (CHOP > 61.8),
# we fade at 1d support/resistance levels. Volume confirms breakout strength.
# Designed to work in both bull and bear markets by adapting to regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high_20 = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_1d.rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range14 = max_high14 - min_low14
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * (np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values) / 
                  np.log10(14)) / np.log10(range14)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if chop[i] < 38.2:  # Trending regime - follow breakouts
            # Long breakout: price breaks above 1d Donchian high + uptrend + volume
            if (close[i] > donchian_high_20_aligned[i] and
                close[i] > ema50_1d_aligned[i] and  # Uptrend filter
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 1d Donchian low + downtrend + volume
            elif (close[i] < donchian_low_20_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and  # Downtrend filter
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                # Hold position in trending regime
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        elif chop[i] > 61.8:  # Ranging regime - fade at extremes
            # Fade long at support: price near 1d Donchian low + oversold bounce
            if (close[i] <= donchian_low_20_aligned[i] * 1.005 and  # Near support
                close[i] > donchian_low_20_aligned[i] and
                close[i] > close[i-1] and  # Price rising
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Fade short at resistance: price near 1d Donchian high + overbought rejection
            elif (close[i] >= donchian_high_20_aligned[i] * 0.995 and  # Near resistance
                  close[i] < donchian_high_20_aligned[i] and
                  close[i] < close[i-1] and  # Price falling
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                # Hold position or flat in ranging regime
                signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
        else:  # Transition regime - reduce activity
            # Only trade with strong alignment
            if (close[i] > donchian_high_20_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                volume_filter[i] and
                chop[i] < chop[i-1]):  # Chop falling = trending emerging
                signals[i] = 0.15
                position = 1
            elif (close[i] < donchian_low_20_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  volume_filter[i] and
                  chop[i] < chop[i-1]):  # Chop falling = trending emerging
                signals[i] = -0.15
                position = -1
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ChopRegime_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0