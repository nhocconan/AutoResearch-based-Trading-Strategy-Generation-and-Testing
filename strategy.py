#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h RSI(14) for momentum direction and 1d volume regime filter.
Long when 4h RSI > 55 AND 1d volume > 1.5x 20-day average volume (high volume regime).
Short when 4h RSI < 45 AND 1d volume > 1.5x 20-day average volume.
Exit when 4h RSI crosses back to neutral (45-55) OR ATR trailing stop (2.0*ATR).
Uses discrete position sizing 0.20 targeting ~20-40 trades/year on 1h timeframe.
Volume regime filter ensures trades only occur during institutional participation,
reducing false signals in low-volume choppy markets. Works in both bull/bear as
it follows momentum with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h RSI(14) for momentum direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:  # Need enough for RSI
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume average
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # ATR(14) for 1h trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 14)  # rsi4h, vol_ma20_1d, atr14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi_4h_aligned[i]
        vol_ma_val = vol_ma_20_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: 4h RSI > 55 AND high volume regime (1d volume > 1.5x 20-day avg)
            if rsi_val > 55 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: 4h RSI < 45 AND high volume regime
            elif rsi_val < 45 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
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
            
            # Primary exit: 4h RSI crosses back to neutral zone (45-55)
            if position == 1 and rsi_val < 55:
                exit_signal = True
            elif position == -1 and rsi_val > 45:
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hRSI_VolumeRegime_ATRTrailingStop"
timeframe = "1h"
leverage = 1.0