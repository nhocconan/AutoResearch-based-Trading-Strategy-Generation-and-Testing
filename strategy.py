#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h range reversion with 1d Bollinger Bands and volume filter
# In ranging markets, price tends to revert to the mean from Bollinger Band extremes
# Long when price touches or breaks below lower BB(20,2) + volume confirmation
# Short when price touches or breaks above upper BB(20,2) + volume confirmation
# Exit when price returns to the 1d SMA(20) mean
# Designed for low frequency (~20-40 trades/year) with edge in ranging markets
# Works in both bull and bear as ranging behavior occurs in all regimes

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands on 1d close
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Calculate volume spike using 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        sma = sma_20_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price at or below lower BB + volume spike
            if price <= lower and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price at or above upper BB + volume spike
            elif price >= upper and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to SMA(20) mean
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to or above the mean
                if price >= sma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to or below the mean
                if price <= sma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BollingerBand_Reversion_Volume"
timeframe = "12h"
leverage = 1.0