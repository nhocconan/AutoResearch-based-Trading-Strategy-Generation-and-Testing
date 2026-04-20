#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX + 1d Volume Spike + 12h Price Momentum
# - ADX(14) > 25 on 12h indicates trending market (avoids chop)
# - Volume spike on 1d (>1.5x 20-period average) confirms institutional interest
# - Price momentum: 12h close > open (bullish) or close < open (bearish)
# - Long when ADX>25, volume spike, and bullish candle
# - Short when ADX>25, volume spike, and bearish candle
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate volume spike on 1d: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate ADX on 12h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Price momentum: bullish/bearish candle
    bullish = close > prices['open'].values
    bearish = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(adx[i]) or np.isnan(volume_spike_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: ADX>25, volume spike, bullish candle
            if adx[i] > 25 and volume_spike_aligned[i] and bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: ADX>25, volume spike, bearish candle
            elif adx[i] > 25 and volume_spike_aligned[i] and bearish[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: ADX drops below 20 or opposite signal
            if adx[i] < 20 or (adx[i] > 25 and volume_spike_aligned[i] and bearish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: ADX drops below 20 or opposite signal
            if adx[i] < 20 or (adx[i] > 25 and volume_spike_aligned[i] and bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ADX_VolumeSpike_Momentum"
timeframe = "12h"
leverage = 1.0