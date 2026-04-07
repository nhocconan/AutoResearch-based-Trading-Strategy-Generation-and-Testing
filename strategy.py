#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour ATR volatility filter and volume confirmation
# Designed for low frequency (target: 25-40 trades/year) to minimize fee drag
# Uses 12h ATR to filter out low-volatility chop and only trade when volatility is elevated
# Volume confirmation ensures breakouts have institutional participation
# Works in both bull and bear markets by capturing strong momentum moves during volatile periods

name = "4h_donchian20_12h_atr_volume_v1"
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
    
    # 12h ATR volatility filter (14-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average
        if i >= 50:
            atr_ma = np.nanmean(atr_12h_aligned[i-50:i]) if i >= 50 else atr_12h_aligned[i]
            vol_filter = atr_12h_aligned[i] > atr_ma
        else:
            vol_filter = False
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1] if i > 0 else False
        breakout_down = close[i] < donchian_low[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on downside breakout
            if breakout_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on upside breakout
            if breakout_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with volatility and volume confirmation
            # Long on upside breakout with volatility and volume
            if breakout_up and vol_filter and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short on downside breakout with volatility and volume
            elif breakout_down and vol_filter and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals