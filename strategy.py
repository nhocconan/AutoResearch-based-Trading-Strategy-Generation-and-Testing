#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d volume confirmation and 1w ADX trend filter
# Long when price breaks above upper BB AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# Short when price breaks below lower BB AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# Exit when price crosses back to middle BB OR 1w ADX drops below 20 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Bollinger Bands provide dynamic volatility-based support/resistance, volume spike confirms conviction,
# 1w ADX ensures we only trade in trending conditions to avoid whipsaws in ranging markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "6h_BB_Breakout_1dVolumeSpike_1wADX_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 6h data (20, 2)
    if len(close) >= 20:
        close_s = pd.Series(close)
        bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
        bb_std = close_s.rolling(window=20, min_periods=20).std().values
        bb_upper = bb_middle + 2.0 * bb_std
        bb_lower = bb_middle - 2.0 * bb_std
    else:
        return np.zeros(n)
    
    # Get 1d volume data for confirmation
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(vol_1d), dtype=bool)
    
    # Align 1d volume spike to 6h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14) on 1w data
    if len(high_1w) >= 14:
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # First TR is NaN
        
        # Directional Movement
        up_move = high_1w[1:] - high_1w[:-1]
        down_move = low_1w[:-1] - low_1w[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        return np.zeros(n)
    
    # ADX trend conditions
    adx_trending = adx > 25
    adx_ranging = adx < 20
    
    # Align 1w ADX conditions to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1w, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1w, adx_ranging.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_ranging_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB AND volume spike AND 1w trending (ADX > 25)
            if (close[i] > bb_upper[i] and 
                volume_spike_aligned[i] > 0.5 and 
                adx_trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB AND volume spike AND 1w trending (ADX > 25)
            elif (close[i] < bb_lower[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  adx_trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to middle BB OR 1w becomes ranging (ADX < 20)
            if (close[i] < bb_middle[i] or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to middle BB OR 1w becomes ranging (ADX < 20)
            if (close[i] > bb_middle[i] or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals