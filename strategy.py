#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume confirmation.
Long when price breaks above R1 AND close > 4h EMA20 AND volume > 1.5x 20-period average.
Short when price breaks below S1 AND close < 4h EMA20 AND volume > 1.5x 20-period average.
Exit when price retouches the Camarilla pivot point (PP) or ATR(14) trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.20) to minimize fee drag. Targets 15-35 trades/year per symbol.
Uses 4h for signal direction (trend filter), 1h only for entry timing precision.
Session filter: 08-20 UTC to avoid low-liquidity hours.
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate Camarilla levels from 1d OHLC (for more stable levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.1 / 4.0
    S1 = PP - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 2)  # EMA20 needs 20, vol MA needs 20, Camarilla needs 2
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(ema20_4h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(PP_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema20_val = ema20_4h_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        PP_val = PP_aligned[i]
        
        if position == 0:
            # Long: Break above R1 AND uptrend (close > 4h EMA20) AND volume confirmation
            if close[i] > R1_val and close[i] > ema20_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
                highest_since_entry = price
            # Short: Break below S1 AND downtrend (close < 4h EMA20) AND volume confirmation
            elif close[i] < S1_val and close[i] < ema20_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Price retouches pivot point (PP)
            if position == 1 and close[i] <= PP_val:
                exit_signal = True
            elif position == -1 and close[i] >= PP_val:
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

name = "1H_Camarilla_R1S1_Breakout_4hEMA20_Trend_VolumeConfirmation_ATRTrailingStop_PPExit"
timeframe = "1h"
leverage = 1.0