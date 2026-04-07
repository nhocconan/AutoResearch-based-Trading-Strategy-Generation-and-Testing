#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian(20) breakout + volume confirmation
# Uses 1d Donchian breakout for directional signal with 4h Choppiness Index to filter ranging markets:
# - Long when price breaks above 1d Donchian(20) high AND 4h Choppiness Index < 40 (trending) AND volume > 20-period average
# - Short when price breaks below 1d Donchian(20) low AND 4h Choppiness Index < 40 (trending) AND volume > 20-period average
# - Exit on opposite Donchian breakout or when Choppiness Index > 60 (ranging)
# - Designed for low frequency (target: 20-40 trades/year) to minimize fee drag
# - Choppiness Index avoids whipsaws in ranging markets; Donchian captures strong momentum moves

name = "4h_chop_regime_1d_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # 4h Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop = np.zeros(n)
    for i in range(n):
        if atr[i] > 0 and hh[i] > ll[i]:
            chop[i] = 100 * np.log10((hh[i] - ll[i]) / (atr[i] * atr_period)) / np.log10(atr_period)
        else:
            chop[i] = 50.0  # Neutral value when calculation not possible
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Regime filter: trending market (Choppiness Index < 40)
        trending = chop[i] < 40
        ranging = chop[i] > 60
        
        # Donchian breakout conditions (using 1d data)
        breakout_up = close[i] > donchian_high_1d_aligned[i-1] if i > 0 else False
        breakout_down = close[i] < donchian_low_1d_aligned[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on downside breakout or ranging market
            if breakout_down or ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on upside breakout or ranging market
            if breakout_up or ranging:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long on upside breakout in trending market
            if breakout_up and trending and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short on downside breakout in trending market
            elif breakout_down and trending and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals