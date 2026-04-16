#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Band squeeze breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 1w ADX > 25 (trending) AND 6h volume > 1.5x 20-period average.
# Short when price breaks below lower BB(20,2) AND 1w ADX > 25 (trending) AND 6h volume > 1.5x 20-period average.
# Exit when price crosses back inside the Bollinger Bands.
# Uses discrete position size 0.25. BB squeeze breakouts capture volatility expansion in trending markets,
# ADX filter ensures we only trade in strong trends, volume confirmation reduces false breakouts.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 6h data once before loop for volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # === 1d Indicators: Bollinger Bands (20,2) ===
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # === 1w Indicators: ADX(14) ===
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # === 6h Indicators: Volume average ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (6h)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper_bb = upper_bb_aligned[i]
        lower_bb = lower_bb_aligned[i]
        adx = adx_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 6h volume aligned
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        vol_6h_current = vol_6h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < upper_bb:  # Exit when price crosses back inside upper BB
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > lower_bb:  # Exit when price crosses back inside lower BB
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper BB AND ADX > 25 (trending) AND 6h volume > 1.5x 20-period avg
            if (price > upper_bb) and (adx > 25) and (vol_6h_current > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower BB AND ADX > 25 (trending) AND 6h volume > 1.5x 20-period avg
            elif (price < lower_bb) and (adx > 25) and (vol_6h_current > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dBB_Squeeze_Breakout_1wADX_TrendFilter_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0