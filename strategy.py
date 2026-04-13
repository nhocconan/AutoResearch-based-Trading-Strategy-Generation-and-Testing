#!/usr/bin/env python3
"""
1h_4h_1D_Camarilla_Breakout_Volume_Confirmation_v1
Hypothesis: Use 4h and daily trends to filter 1h breakouts. Long when 1h price breaks above daily Camarilla H4 with volume > 1.8x 50-period average, price > 4h EMA20, and 4h close > daily EMA50. Short when 1h price breaks below daily Camarilla L4 with volume confirmation, price < 4h EMA20, and 4h close < daily EMA50. This multi-timeframe approach reduces false signals and adapts to bull/bear markets by requiring trend alignment. Target 15-37 trades/year via strict conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.8x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean()
    volume_expansion = volume > (vol_ma_50 * 1.8)
    
    # 4h EMA20 for 1h entry filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema20_4h_raw = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h_raw)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema50_1d_raw = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_raw)
    
    # Previous day's high/low/close for Camarilla calculation
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    camarilla_h4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Align daily levels to 1h timeframe (wait for daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(60, n):  # warmup period
        # Skip if any required data is not ready or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_expansion[i]) or not in_session[i]):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long signal: 1h break above daily Camarilla H4 with volume expansion, price > 4h EMA20, and 4h close > daily EMA50
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      close[i] > ema20_4h_aligned[i] and 
                      df_4h['close'].values[i // 4] > ema50_1d_raw[i // 24]) if i // 4 < len(df_4h) and i // 24 < len(df_1d) else False
        
        # Short signal: 1h break below daily Camarilla L4 with volume expansion, price < 4h EMA20, and 4h close < daily EMA50
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       close[i] < ema20_4h_aligned[i] and 
                       df_4h['close'].values[i // 4] < ema50_1d_raw[i // 24]) if i // 4 < len(df_4h) and i // 24 < len(df_1d) else False
        
        # Exit conditions: minimum holding period reached (4 hours) or opposite signal
        if position == 1 and (bars_since_entry >= 4 or short_signal):
            position = -1 if short_signal else 0
            signals[i] = -position_size if short_signal else 0.0
            bars_since_entry = 0
        elif position == -1 and (bars_since_entry >= 4 or long_signal):
            position = 1 if long_signal else 0
            signals[i] = position_size if long_signal else 0.0
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_1D_Camarilla_Breakout_Volume_Confirmation_v1"
timeframe = "1h"
leverage = 1.0