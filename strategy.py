#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
Long when price breaks above Donchian upper band and ATR(1d) > ATR_ma(1d,20) (high volatility regime) with volume > 2.0x average.
Short when price breaks below Donchian lower band and ATR(1d) > ATR_ma(1d,20) with volume > 2.0x average.
Exit on opposite Donchian break or ATR regime shift to low volatility.
Uses 4h timeframe targeting 75-200 total trades over 4 years.
Donchian channels provide clear breakout levels, 1d ATR filter ensures we only trade in high volatility regimes,
volume spike confirms breakout strength. Designed to capture strong momentum moves while avoiding low volatility chop.
Works in both bull (breakouts up) and bear (breakouts down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(1d) - True Range then smoothed
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first period has no TR
    
    atr_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # ATR moving average for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Donchian(20) on primary timeframe
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_1d_aligned[i]
        atr_ma_val = atr_ma_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Regime filter: only trade in high volatility (ATR > ATR_MA)
        high_vol_regime = atr_val > atr_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND high vol regime AND volume spike
            if (price > donch_high[i] and high_vol_regime and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower band AND high vol regime AND volume spike
            elif (price < donch_low[i] and high_vol_regime and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR low volatility regime
                if (price < donch_low[i] or not high_vol_regime):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR low volatility regime
                if (price > donch_high[i] or not high_vol_regime):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0