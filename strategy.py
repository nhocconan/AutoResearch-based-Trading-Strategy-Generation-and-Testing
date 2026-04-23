#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 12h close > 12h EMA50 AND 6h volume > 2.0x 20-period average volume.
Short when price breaks below Camarilla S3 AND 12h close < 12h EMA50 AND 6h volume > 2.0x 20-period average volume.
Exit when price reaches Camarilla pivot point (PP) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting 12-37 trades/year on 6h timeframe.
Combines price structure (Camarilla pivots), trend filter (12h EMA50), and volume confirmation for robustness across bull/bear regimes.
Camarilla levels calculated from prior completed 12h bar, ensuring no look-ahead bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior completed 12h bar
    df_12h_complete = get_htf_data(prices, '12h')
    if len(df_12h_complete) < 2:
        return np.zeros(n)
    
    high_12h = df_12h_complete['high'].values
    low_12h = df_12h_complete['low'].values
    close_12h = df_12h_complete['close'].values
    
    # Camarilla levels for each 12h bar: based on prior 12h bar's range
    PP = (high_12h + low_12h + close_12h) / 3
    R1 = close_12h + (high_12h - low_12h) * 1.1 / 12
    S1 = close_12h - (high_12h - low_12h) * 1.1 / 12
    R2 = close_12h + (high_12h - low_12h) * 1.1 / 6
    S2 = close_12h - (high_12h - low_12h) * 1.1 / 6
    R3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    S3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    R4 = close_12h + (high_12h - low_12h) * 1.1 / 2
    S4 = close_12h - (high_12h - low_12h) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (using prior completed 12h bar)
    PP_aligned = align_htf_to_ltf(prices, df_12h_complete, PP)
    R3_aligned = align_htf_to_ltf(prices, df_12h_complete, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h_complete, S3)
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
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
    start_idx = max(20, 50)  # volume MA20 and EMA50 need 20 and 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_val = ema_50_12h_aligned[i]
        PP_val = PP_aligned[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND bullish trend (12h close > EMA50) AND volume spike
            if close[i] > R3_val and ema_50_12h_aligned[i] > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND bearish trend (12h close < EMA50) AND volume spike
            elif close[i] < S3_val and ema_50_12h_aligned[i] < ema_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price reaches Camarilla pivot point (PP)
            if position == 1 and close[i] >= PP_val:
                exit_signal = True
            elif position == -1 and close[i] <= PP_val:
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

name = "6H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_PPExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0