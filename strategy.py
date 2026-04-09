#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h ATR regime filter
# Uses 4h Donchian channel breakouts confirmed by volume spike (>2.0x 20-period avg volume)
# Only takes breakouts when 12h ATR(14) is below its 50-period MA (low volatility regime)
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
# Works in both bull/bear: 12h ATR regime filter ensures we trade breakouts only in low volatility environments where they are more reliable

name = "4h_12h_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ATR regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr_12h = np.full(len(df_12h), np.nan)
    atr_12h = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        tr = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
        tr_12h[i] = tr
    
    # Calculate ATR with Wilder's smoothing
    for i in range(len(df_12h)):
        if i < 14:
            atr_12h[i] = np.nan
        elif i == 14:
            atr_12h[i] = np.nanmean(tr_12h[1:15])
        else:
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 50-period MA of ATR for regime filter
    atr_ma_50 = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        if i < 50:
            atr_ma_50[i] = np.nan
        else:
            atr_ma_50[i] = np.mean(atr_12h[i-50:i])
    
    # Align 12h indicators to 4h timeframe
    atr_ma_50_4h = align_htf_to_ltf(prices, df_12h, atr_ma_50)
    atr_4h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 20-period Donchian channels on 4h
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
            np.isnan(atr_ma_50_4h[i]) or 
            np.isnan(atr_4h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_4h[i] < atr_ma_50_4h[i]
        
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