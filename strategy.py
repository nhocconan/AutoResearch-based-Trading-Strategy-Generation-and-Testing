#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian high + ATR(14) > 1.5x ATR(50) (expanding volatility) + volume > 1.3x 20-period avg
# Short when price breaks below 20-period Donchian low + ATR(14) > 1.5x ATR(50) + volume > 1.3x 20-period avg
# Uses 1d Donchian levels and ATR calculated from prior 1d OHLC, aligned to 12h bars
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Donchian breakouts capture strong trends; ATR filter ensures we trade during expanding volatility regimes
# Volume confirmation reduces false breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Donchian and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian levels: 20-period high and low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for 1d bar to close)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === 1d Indicator: ATR Ratio (14/50) for volatility expansion ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = high_1d[0] - close_1d[0]  # first period
    tr3[0] = low_1d[0] - close_1d[0]   # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term > 1.5 indicates expanding volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0.0)
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d data for Donchian(20) + ATR(14,50) + volume(20) + buffer
    warmup = 70
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Volatility filter: only trade when ATR ratio > 1.5 (expanding volatility)
        vol_expanding = atr_ratio_aligned[i] > 1.5
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. ATR ratio > 1.5 (expanding volatility)
        # 3. Volume confirmation
        if (close[i] > high_20_aligned[i]) and \
           vol_expanding and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. ATR ratio > 1.5 (expanding volatility)
        # 3. Volume confirmation
        elif (close[i] < low_20_aligned[i]) and \
             vol_expanding and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR_Ratio_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0