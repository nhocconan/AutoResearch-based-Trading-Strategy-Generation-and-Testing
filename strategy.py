#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above 20-period 12h high, 1d ATR ratio > 1.2 (expanding volatility), and volume > 1.8x 20-bar avg.
# Short when price breaks below 20-period 12h low, 1d ATR ratio > 1.2, and volume > 1.8x 20-bar avg.
# Exit when price reverts to the 12h 20-period midpoint (mean reversion).
# Uses 12h timeframe for lower trade frequency (target: 12-37 trades/year) to minimize fee drag.
# 1d ATR ratio ensures we only trade during expanding volatility regimes, reducing whipsaws.
# Volume confirmation filters low-conviction breakouts.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns with volatility expansion.

name = "12h_Donchian20_1dATRratio_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period average for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / atr_ma_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ATR MA and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_atr_ratio = atr_ratio_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_donchian_mid = donchian_mid[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Volatility regime filter: only trade when ATR ratio > 1.2 (expanding volatility)
        volatility_filter = curr_atr_ratio > 1.2
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, expanding volatility, volume spike
            if (curr_close > curr_donchian_high and 
                volatility_filter and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, expanding volatility, volume spike
            elif (curr_close < curr_donchian_low and 
                  volatility_filter and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to Donchian midpoint (mean reversion)
            if curr_close <= curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to Donchian midpoint (mean reversion)
            if curr_close >= curr_donchian_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals