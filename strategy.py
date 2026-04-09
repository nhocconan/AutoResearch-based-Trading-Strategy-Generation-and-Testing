#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ATR regime filter
# Uses 12h Donchian channel breakouts confirmed by volume spike (>2.0x 20-period avg volume)
# Only takes breakouts when 1d ATR(14) is below its 50-period MA (low volatility regime)
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag
# Works in both bull/bear: 1d ATR regime filter ensures we trade breakouts only in low volatility environments where they are more reliable

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.full(len(df_1d), np.nan)
    atr_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        tr_1d[i] = tr
    
    # Calculate ATR with Wilder's smoothing
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        elif i == 14:
            atr_1d[i] = np.nanmean(tr_1d[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 50-period MA of ATR for regime filter
    atr_ma_50 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 50:
            atr_ma_50[i] = np.nan
        else:
            atr_ma_50[i] = np.mean(atr_1d[i-50:i])
    
    # Align 1d indicators to 12h timeframe
    atr_ma_50_12h = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_ma_50_12h[i]) or 
            np.isnan(atr_12h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_12h[i] < atr_ma_50_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low OR ATR regime turns unfavorable
            if close[i] < donchian_low[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high OR ATR regime turns unfavorable
            if close[i] > donchian_high[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Donchian high
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals