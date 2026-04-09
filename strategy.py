#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and 1d ATR regime filter
# Uses Camarilla levels from 1d for structure, breaks above H3 or below L3 on 4h for entries
# Only takes breakouts when 1d ATR(14) is below its 50-period MA (low volatility regime) for reliability
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
# Works in both bull/bear: 1d ATR regime filter ensures we trade breakouts only in low volatility environments where they are more reliable

name = "4h_1d_camarilla_volume_atr_v1"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
        else:
            # Use previous day's OHLC to avoid look-ahead
            high_prev = df_1d['high'].iloc[i-1]
            low_prev = df_1d['low'].iloc[i-1]
            close_prev = df_1d['close'].iloc[i-1]
            camarilla_h3[i] = close_prev + 1.1 * (high_prev - low_prev) / 2
            camarilla_l3[i] = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Calculate 1d ATR(14) for regime filter
    tr_1d = np.full(len(df_1d), np.nan)
    atr_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
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
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Align 1d ATR regime to 4h timeframe
    atr_ma_50_4h = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
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
        if (np.isnan(camarilla_h3_4h[i]) or 
            np.isnan(camarilla_l3_4h[i]) or 
            np.isnan(atr_ma_50_4h[i]) or 
            np.isnan(atr_4h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # ATR regime filter: only trade when current ATR < ATR MA (low volatility regime)
        atr_regime = atr_4h[i] < atr_ma_50_4h[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Camarilla L3 OR ATR regime turns unfavorable
            if close[i] < camarilla_l3_4h[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla H3 OR ATR regime turns unfavorable
            if close[i] > camarilla_h3_4h[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Camarilla H3
                if close[i] > camarilla_h3_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Camarilla L3
                elif close[i] < camarilla_l3_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals