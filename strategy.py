#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H4, L4, H3, L3) act as key support/resistance. 
# Breakouts above H4 or below L4 with 1d volume > 2x average and ATR > 0.015 (avoiding low volatility) 
# capture institutional moves. Works in both bull and bear markets by trading breakouts in the direction 
# of volatility expansion, avoiding false signals in low-volatility chop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = high_1d[0]
    tr3[0] = low_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Camarilla pivot levels from previous day
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #            H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # We use previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first bar uses current
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_height = 1.5 * (prev_high - prev_low)
    camarilla_h4 = prev_close + camarilla_height
    camarilla_l4 = prev_close - camarilla_height
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 4h volume > 2x 1d average volume
        vol_confirmed = volume[i] > (2.0 * vol_avg_1d_aligned[i])
        
        # Volatility filter: 1d ATR > 0.015 (1.5%) to avoid low volatility chop
        vol_filter = atr_1d_aligned[i] > 0.015
        
        # Entry conditions
        # Long: Close > H4 + volume confirmation + volatility filter
        if vol_confirmed and vol_filter and close[i] > camarilla_h4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Close < L4 + volume confirmation + volatility filter
        elif vol_confirmed and vol_filter and close[i] < camarilla_l4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: reversal signal or volatility collapse
        elif position == 1 and (close[i] < camarilla_l4_aligned[i] or atr_1d_aligned[i] < 0.01):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_h4_aligned[i] or atr_1d_aligned[i] < 0.01):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals