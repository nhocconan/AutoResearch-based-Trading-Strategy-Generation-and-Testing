#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volume confirmation and 1d trend filter
# Long when price breaks above Donchian upper band AND volume > 1.5x ATR-scaled average AND 1d close > 1d EMA50 (uptrend)
# Short when price breaks below Donchian lower band AND volume > 1.5x ATR-scaled average AND 1d close < 1d EMA50 (downtrend)
# Exit when price crosses back to Donchian midpoint OR 1d trend flips
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Donchian provides clear structure, ATR-scaled volume avoids low-vol false breakouts, 1d EMA50 filters trend direction.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_Donchian20_ATRVol_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50
    downtrend_1d = close_1d < ema_50
    
    # Calculate 1d ATR(14) for volume scaling
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr_scaled_vol_ma = vol_ma_20 * (atr_14[-1] / atr_14) if len(atr_14) == len(volume) else vol_ma_20 * 1.0
    # Align ATR to volume length if needed
    if len(atr_14) != len(volume):
        atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
        atr_scaled_vol_ma = vol_ma_20 * (atr_14_aligned[-1] / atr_14_aligned)
    
    # Volume filter: volume > 1.5x ATR-scaled 20-period average volume
    volume_filter = volume > (1.5 * atr_scaled_vol_ma)
    
    # Calculate Donchian channels on 4h data
    lookback = 20
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (upper_band + lower_band) / 2
    
    # Align 1d indicators to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float)) if len(df_1d) < len(prices) else volume_filter
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Recalculate ATR-scaled volume with aligned ATR
    if len(atr_14_aligned) == len(volume):
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        atr_scaled_vol_ma = vol_ma_20 * (atr_14_aligned[-1] / atr_14_aligned)
        volume_filter = volume > (1.5 * atr_scaled_vol_ma)
        volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float)) if len(df_1d) < len(prices) else volume_filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper band AND volume spike AND 1d uptrend
            if (close[i] > upper_band[i] and 
                volume_filter_aligned[i] > 0.5 and 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band AND volume spike AND 1d downtrend
            elif (close[i] < lower_band[i] and 
                  volume_filter_aligned[i] > 0.5 and 
                  downtrend_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR 1d trend flips to downtrend
            if (close[i] < midpoint[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR 1d trend flips to uptrend
            if (close[i] > midpoint[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals