#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_Regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for previous day
    # Using previous day's high, low, close to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    # First day: use same values (will be filtered out by warmup)
    phigh[0] = high_1d[0]
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    # Calculate pivot and ranges
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Camarilla levels
    R1 = pclose + (range_ * 1.1 / 12)
    S1 = pclose - (range_ * 1.1 / 12)
    
    # Calculate volume spike (volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Calculate 4h ADX for trend filter (avoid choppy markets)
    # ADX > 25 indicates trending market
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 100 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr * 100 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        # ADX filter: only trade in trending markets (ADX > 25)
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long when price breaks above R1 with volume spike
            if (close[i] > R1_aligned[i] and 
                volume_spike_aligned[i] and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume spike
            elif (close[i] < S1_aligned[i] and 
                  volume_spike_aligned[i] and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price drops below S1 or ADX drops below 20 (trend weakening)
            if (close[i] < S1_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 or ADX drops below 20
            if (close[i] > R1_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals