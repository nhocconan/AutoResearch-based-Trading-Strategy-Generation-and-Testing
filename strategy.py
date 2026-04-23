#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 breakout with 1d Williams %R oversold/overbought filter and volume confirmation.
Long when price breaks above Camarilla R4 AND 1d Williams %R < -80 (oversold) AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S4 AND 1d Williams %R > -20 (overbought) AND volume > 1.8x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) OR ATR trailing stop (2.0*ATR from extreme).
Williams %R from daily timeframe provides reversal timing edge in ranging markets, effective in BOTH bull and bear regimes.
6h timeframe balances noise and trade frequency to target 12-37 trades/year, avoiding fee drag while capturing meaningful swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period) for reversal timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
                          -50.0)  # neutral when range is zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Camarilla levels from daily timeframe (using previous day's data)
    rng = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s4 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(10) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 10)  # Williams %R needs 14, vol MA needs 20, ATR needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        williams_val = williams_r_aligned[i]
        pp_val = pp_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R4 AND oversold (Williams %R < -80) AND volume spike (1.8x avg)
            if close[i] > r4_val and williams_val < -80 and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S4 AND overbought (Williams %R > -20) AND volume spike (1.8x avg)
            elif close[i] < s4_val and williams_val > -20 and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla pivot point (PP)
            if position == 1 and close[i] <= pp_val:
                exit_signal = True
            elif position == -1 and close[i] >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4S4_Breakout_1dWilliamsR_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0