#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme reversal with 1d volume confirmation and 1d chop regime filter
    # Long: Williams %R(14) < -80 (oversold) AND volume > 1.5x 20-period average AND chop > 61.8 (ranging market)
    # Short: Williams %R(14) > -20 (overbought) AND volume > 1.5x 20-period average AND chop > 61.8
    # Exit: Williams %R crosses above -50 (long exit) or below -50 (short exit)
    # Using 12h timeframe for optimal trade frequency (target 12-37/year), Williams %R for mean reversion in ranging markets,
    # Volume confirmation to avoid false signals, and chop filter to ensure we only trade in ranging conditions.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R, volume, and chop calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Williams %R(14)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align daily Williams %R to 12h
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate daily volume confirmation (>1.5x 20-period average)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (1.5 * vol_ma)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate daily choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low)) / log10(n)
    tr1 = pd.Series(df_1d['high']).rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0], raw=False)
    tr2 = np.abs(pd.Series(df_1d['high']).rolling(window=2).apply(lambda x: x.iloc[1] - df_1d['close'].iloc[x.index[0]], raw=False))
    tr3 = np.abs(pd.Series(df_1d['low']).rolling(window=2).apply(lambda x: x.iloc[1] - df_1d['close'].iloc[x.index[0]], raw=False))
    # Simplified TR calculation
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if atr[i] > 0 and (highest_high_14[i] - lowest_low_14[i]) > 0:
            sum_atr = np.sum(atr[i-13:i+1])  # 14-period sum
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10((highest_high_14[i] - lowest_low_14[i]) / atr[i])
        else:
            chop[i] = 50  # neutral value
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop_1d_aligned[i] > 61.8
        
        # Williams %R extreme conditions
        williams_oversold = williams_r_1d_aligned[i] < -80
        williams_overbought = williams_r_1d_aligned[i] > -20
        
        # Exit conditions: Williams %R crosses above/below -50
        williams_long_exit = williams_r_1d_aligned[i] > -50
        williams_short_exit = williams_r_1d_aligned[i] < -50
        
        # Entry logic: Williams extreme + volume confirmation + ranging market
        long_entry = williams_oversold and volume_spike_1d_aligned[i] and ranging_market
        short_entry = williams_overbought and volume_spike_1d_aligned[i] and ranging_market
        
        # Exit logic: Williams %R crosses -50
        long_exit = williams_long_exit
        short_exit = williams_short_exit
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williamsr_extreme_volume_chop_v1"
timeframe = "12h"
leverage = 1.0