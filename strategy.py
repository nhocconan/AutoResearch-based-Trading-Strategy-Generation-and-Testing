#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ATR filter
# Donchian breakout captures momentum in both bull/bear markets
# Volume spike confirms institutional participation
# ATR filter avoids low-volatility whipsaws
# Position size 0.25 to limit drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period SMA)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-20:i])
    
    # Volume spike threshold: current volume > 2.0 x 20-period average
    vol_spike = np.zeros(len(df_1d))
    for i in range(20, len(df_1d)):
        if vol_ma_20[i] > 0:
            vol_spike[i] = vol_1d[i] > (2.0 * vol_ma_20[i])
    
    # Align 1d volume spike to 4h
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate ATR(14) for 4h
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr0 = high[i] - low[i]
        tr1 = abs(high[i] - close[i-1])
        tr2 = abs(low[i] - close[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: avoid low volatility (ATR < 0.5% of price)
        if atr[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout + volume spike
            breakout_long = close[i] > donchian_high[i] and vol_spike_4h[i]
            breakout_short = close[i] < donchian_low[i] and vol_spike_4h[i]
            
            if breakout_long:
                position = 1
                signals[i] = 0.25
            elif breakout_short:
                position = -1
                signals[i] = -0.25
    
    return signals