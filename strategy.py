#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; ADX filters for trending vs ranging markets.
# In ranging markets (ADX < 25), we fade extremes; in trending markets (ADX >= 25), we follow trend.
# Volume spike (>1.5x 20-period average) confirms momentum. Designed for low trade frequency (~15-30/year).
# Works in both bull and bear markets by adapting to regime via ADX.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    atr_14 = np.zeros(len(high_1d))
    for i in range(14, len(high_1d)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    plus_di_14 = np.zeros(len(high_1d))
    minus_di_14 = np.zeros(len(high_1d))
    dx = np.zeros(len(high_1d))
    for i in range(14, len(high_1d)):
        if atr_14[i] != 0:
            plus_di_14[i] = 100 * np.mean(plus_dm[i-13:i+1]) / atr_14[i]
            minus_di_14[i] = 100 * np.mean(minus_dm[i-13:i+1]) / atr_14[i]
            if plus_di_14[i] + minus_di_14[i] != 0:
                dx[i] = 100 * abs(plus_di_14[i] - minus_di_14[i]) / (plus_di_14[i] + minus_di_14[i])
    
    adx_14 = np.zeros(len(high_1d))
    for i in range(28, len(high_1d)):
        adx_14[i] = np.mean(dx[i-13:i+1])
    
    # Calculate 14-period Williams %R on 12h
    highest_high_14 = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    close = prices['close'].values
    willr = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Align 1d ADX to 12h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or 
            np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr = willr[i]
        adx = adx_14_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Ranging market (ADX < 25): mean reversion at extremes
            if adx < 25:
                # Long when oversold (-80 to -100) with volume spike
                if wr <= -80 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short when overbought (0 to -20) with volume spike
                elif wr >= -20 and vol_spike:
                    signals[i] = -0.25
                    position = -1
            # Trending market (ADX >= 25): follow trend
            else:
                # Long when not overbought and rising from oversold
                if wr > -50 and wr < -20 and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short when not oversold and falling from overbought
                elif wr < -50 and wr > -80 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when overbought or trend weakness
                if wr >= -20 or (adx < 20 and wr > -50):
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when oversold or trend weakness
                if wr <= -80 or (adx < 20 and wr < -50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_ADX25_Volume"
timeframe = "12h"
leverage = 1.0