#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + volume confirmation + 1d ATR regime filter
# Uses 1d Camarilla pivot levels (H3/L3) for breakout signals, confirmed by volume spike (>2.0x 20-period avg volume)
# Only takes breakouts when 1d ATR(14) is below its 50-period MA (low volatility regime) to avoid false breakouts in high volatility
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fee drag
# Works in both bull/bear: 1d ATR regime filter ensures we trade breakouts only in low volatility environments where they are more reliable

name = "12h_1d_camarilla_vol_atr_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivots_h3 = np.full(len(df_1d), np.nan)
    pivots_l3 = np.full(len(df_1d), np.nan)
    pivots_h4 = np.full(len(df_1d), np.nan)
    pivots_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            pivots_h3[i] = np.nan
            pivots_l3[i] = np.nan
            pivots_h4[i] = np.nan
            pivots_l4[i] = np.nan
        else:
            # Calculate pivot point (PP) = (H + L + C) / 3
            pp = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
            # Calculate range = H - L
            rng = high_1d[i] - low_1d[i]
            # Camarilla levels
            pivots_h3[i] = pp + rng * 1.1 / 4
            pivots_l3[i] = pp - rng * 1.1 / 4
            pivots_h4[i] = pp + rng * 1.1 / 2
            pivots_l4[i] = pp - rng * 1.1 / 2
    
    # Calculate 1d ATR(14) for regime filter
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
    pivots_h3_12h = align_htf_to_ltf(prices, df_1d, pivots_h3)
    pivots_l3_12h = align_htf_to_ltf(prices, df_1d, pivots_l3)
    pivots_h4_12h = align_htf_to_ltf(prices, df_1d, pivots_h4)
    pivots_l4_12h = align_htf_to_ltf(prices, df_1d, pivots_l4)
    atr_ma_50_12h = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
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
        if (np.isnan(pivots_h3_12h[i]) or 
            np.isnan(pivots_l3_12h[i]) or 
            np.isnan(pivots_h4_12h[i]) or 
            np.isnan(pivots_l4_12h[i]) or 
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
            # Exit conditions: price closes below Camarilla L3 OR ATR regime turns unfavorable
            if close[i] < pivots_l3_12h[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Camarilla H3 OR ATR regime turns unfavorable
            if close[i] > pivots_h3_12h[i] or not atr_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla breakout with volume confirmation and ATR regime filter
            if volume_confirm and atr_regime:
                # Long breakout: price closes above Camarilla H3
                if close[i] > pivots_h3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Camarilla L3
                elif close[i] < pivots_l3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals