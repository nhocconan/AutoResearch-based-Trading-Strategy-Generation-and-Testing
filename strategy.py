#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation.
Long when price breaks above 20-period Donchian high with 1d ATR > 20-period 1d ATR MA and volume > 1.5x 20-period 4h volume MA.
Short when price breaks below 20-period Donchian low with same volatility and volume filters.
Exit when price returns to the 20-period Donchian midpoint or reverses with volume confirmation.
Uses 1d for volatility regime (avoid choppy low-vol periods) and 4h for execution.
Designed to capture strong breakouts during expanding volatility in both bull and bear markets.
Volatility filter ensures trades occur only when ATR is elevated, reducing false breakouts in ranging markets.
Target: 20-40 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR(20)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period has no previous close
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR MA(20) for volatility regime filter
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 4h volume MA(20) for volume confirmation
    volume_series = pd.Series(volume)
    vol_ma_20_4h = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # need enough for Donchian(20) and ATR calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_20_1d_aligned[i]) or 
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: 1d ATR > 20-period 1d ATR MA (expanding volatility)
        vol_regime = atr_1d_aligned[i] > atr_ma_20_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-period MA
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volatility and volume confirmation
            if (close[i] > donch_high[i] and 
                vol_regime and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volatility and volume confirmation
            elif (close[i] < donch_low[i] and 
                  vol_regime and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR breaks below Donchian low with volume (reversal)
            if (close[i] <= donch_mid[i] or 
                (close[i] < donch_low[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR breaks above Donchian high with volume (reversal)
            if (close[i] >= donch_mid[i] or 
                (close[i] > donch_high[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRVolRegime_VolumeConf"
timeframe = "4h"
leverage = 1.0