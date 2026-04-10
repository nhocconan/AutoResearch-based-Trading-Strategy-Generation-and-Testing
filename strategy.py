#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with weekly ATR filter and volume confirmation
# - Long when price breaks above H4 Camarilla resistance AND ATR(14,1w) > ATR(50,1w) (expanding volatility)
# - Short when price breaks below L4 Camarilla support AND ATR(14,1w) > ATR(50,1w)
# - Volume confirmation: 12h volume > 1.5x 20-period 12h volume SMA
# - Exit: price reverts to H3/L3 levels or opposite breakout with volume
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Uses weekly ATR filter to ensure trades occur during volatile regimes, reducing whipsaw
# - Camarilla pivot levels from 1d provide institutional structure that works in both bull/bear

name = "12h_camarilla_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h ATR(14) and ATR(50) for volatility regime filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry price for stoploss logic
    entry_price = np.full(n, np.nan)
    
    # Load weekly HTF data ONCE before loop (as per rules)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly ATR for regime filter
    tr1w = pd.Series(df_1w['high'] - df_1w['low'])
    tr2w = pd.Series(np.abs(df_1w['high'] - np.roll(df_1w['close'].values, 1)))
    tr3w = pd.Series(np.abs(df_1w['low'] - np.roll(df_1w['close'].values, 1)))
    trw = pd.concat([tr1w, tr2w, tr3w], axis=1).max(axis=1)
    atr_14_1w = pd.Series(trw).rolling(window=14, min_periods=14).mean().values
    atr_50_1w = pd.Series(trw).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly ATR to 12h timeframe (properly delayed for completed weekly bars)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    atr_50_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_50_1w)
    
    # Load daily HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_high = (high_1d - low_1d)
    camarilla_H4 = close_1d + 1.1 * camarilla_high / 2.0
    camarilla_L4 = close_1d - 1.1 * camarilla_high / 2.0
    camarilla_H3 = close_1d + 1.1 * camarilla_high / 4.0
    camarilla_L3 = close_1d - 1.1 * camarilla_high / 4.0
    
    # Align daily Camarilla levels to 12h timeframe (properly delayed for completed daily bars)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(volume_sma_20[i]) or
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(atr_50_1w_aligned[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly volatility filter: ATR(14,1w) > ATR(50,1w) (expanding volatility regime)
        vol_regime = atr_14_1w_aligned[i] > atr_50_1w_aligned[i]
        
        # Volume confirmation: 12h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        if position == 0:  # Flat - look for entry
            # Long when price breaks above H4 resistance with volume and volatility
            if close[i] > camarilla_H4_aligned[i] and vol_regime and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            # Short when price breaks below L4 support with volume and volatility
            elif close[i] < camarilla_L4_aligned[i] and vol_regime and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit when price reverts to H3 level or breaks below L4 with volume
            exit_condition = (close[i] < camarilla_H3_aligned[i]) or \
                           (close[i] < camarilla_L4_aligned[i] and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit when price reverts to L3 level or breaks above H4 with volume
            exit_condition = (close[i] > camarilla_L3_aligned[i]) or \
                           (close[i] > camarilla_H4_aligned[i] and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals