#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Uses 1d ATR(14) to filter low volatility regimes and avoid whipsaws
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
# Works in both bull and bear markets by using Donchian channels for structure and ATR for volatility regime filtering
# Prioritizes BTC/ETH performance with SOL as secondary

name = "12h_Donchian20_Volume_1dATR14_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (EMA with alpha=1/14)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels on 12h data (20-period)
    # Upper channel: highest high of last 20 periods
    # Lower channel: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (moderate to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # ATR filter: only trade when volatility is above average (avoid low volatility chop)
        # Use 1d ATR compared to its 50-period EMA to detect elevated volatility regimes
        if i >= 150:  # Need enough history for ATR EMA
            atr_ema_50 = pd.Series(atr_14_1d_aligned[:i+1]).ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
            vol_filter = atr_14_1d_aligned[i] > atr_ema_50
        else:
            vol_filter = True  # Allow trading during warmup period
        
        # Donchian breakout with volume and volatility filters
        # Long: Price breaks above Donchian upper channel + volume spike + volatility filter
        # Short: Price breaks below Donchian lower channel + volume spike + volatility filter
        if position == 0:
            if (close[i] > donchian_upper[i] and volume_spike and vol_filter):
                signals[i] = 0.25
                position = 1
            elif (close[i] < donchian_lower[i] and volume_spike and vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian lower channel OR volatility drops below average
            if close[i] < donchian_lower[i] or (i >= 150 and atr_14_1d_aligned[i] <= atr_ema_50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian upper channel OR volatility drops below average
            if close[i] > donchian_upper[i] or (i >= 150 and atr_14_1d_aligned[i] <= atr_ema_50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals