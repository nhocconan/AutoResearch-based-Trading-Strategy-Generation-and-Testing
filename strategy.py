#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) Breakout with 1d Volume Confirmation and Choppiness Regime Filter
# Hypothesis: Donchian breakouts capture trend continuation moves; daily volume confirms institutional participation.
# Choppiness filter avoids whipsaws in ranging markets. Works in bull via upper band breakouts, in bear via lower band breakdowns.
# Target: 25-40 trades/year to minimize fee drag.
name = "4h_donchian20_1d_volume_chop_v1"
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
    
    # Get 1d data for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d choppiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = []
    tr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
        if i < 14:
            atr_1d.append(np.nan)
        else:
            if i == 14:
                atr = np.mean(tr_1d[0:15])
            else:
                atr = (atr_1d[-1] * 13 + tr_1d[i]) / 14
            atr_1d.append(atr)
    atr_1d = np.array(atr_1d)
    highest_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10((atr_1d * 14) / (highest_1d - lowest_1d)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Donchian Channels (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1d average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        # Choppiness filter: avoid ranging markets (chop > 61.8)
        chop_filter = chop_1d_aligned[i] <= 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (trend reversal)
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (trend reversal)
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above Donchian high + volume confirmation + chop filter
            if close[i] > donch_high[i] and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low + volume confirmation + chop filter
            elif close[i] < donch_low[i] and vol_confirm and chop_filter:
                position = -1
                signals[i] = -0.25
    
    return signals