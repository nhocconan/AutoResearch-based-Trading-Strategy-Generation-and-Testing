#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 of daily close for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe (use previous day's values)
    bull_power_1d_prev = np.roll(bull_power_1d, 1)
    bear_power_1d_prev = np.roll(bear_power_1d, 1)
    bull_power_1d_prev[0] = np.nan
    bear_power_1d_prev[0] = np.nan
    
    bull_power = align_htf_to_ltf(prices, df_1d, bull_power_1d_prev)
    bear_power = align_htf_to_ltf(prices, df_1d, bear_power_1d_prev)
    
    # Volume confirmation: current volume > 1.5 * 6-period average (6h * 6 = 36h)
    volume_ma6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma8 = pd.Series(atr).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need Elder Ray and ATR MA8
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma6[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma8[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 6-period average
        volume_filter = volume[i] > (1.5 * volume_ma6[i])
        # Volatility filter: ATR > ATR MA8 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma8[i]
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) with volume and volatility
            if bull_power[i] > 0 and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling pressure) with volume and volatility
            elif bear_power[i] < 0 and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 (weakening buying pressure) or volatility drops
            if bull_power[i] <= 0 or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 (weakening selling pressure) or volatility drops
            if bear_power[i] >= 0 or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_Volume"
timeframe = "6h"
leverage = 1.0