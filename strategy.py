#!/usr/bin/env python3
"""
4h_12h_1d_Camarilla_Pullback_Strategy
Hypothesis: On 4h timeframe, enters long when price pulls back to L3 after breaking above H3 (bullish), 
and short when price pulls back to H3 after breaking below L3 (bearish), with volume confirmation. 
Uses 12h trend (EMA21) for direction filter and 1d volatility (ATR14) to avoid choppy markets. 
Designed to work in both bull and bear markets by requiring pullbacks within established trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels for previous 1d bar
    hl_range_1d = high_1d - low_1d
    H3 = close_1d + 1.100 * hl_range_1d  # H3 level
    L3 = close_1d - 1.100 * hl_range_1d  # L3 level
    H4 = close_1d + 1.125 * hl_range_1d  # Stop level
    L4 = close_1d - 1.125 * hl_range_1d  # Stop level
    
    # Get 12h data for trend filter (EMA21)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all 1d signals to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Align 12h EMA to 4h timeframe
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate volume spike (2x 20-period average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(ema21_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility (potential chop)
        vol_filter = atr14_aligned[i] < (np.mean(atr14_aligned[max(0, i-50):i+1]) * 2.0)
        
        # Trend filter: price above/below 12h EMA21
        uptrend = close[i] > ema21_12h_aligned[i]
        downtrend = close[i] < ema21_12h_aligned[i]
        
        # Pullback conditions: price returns to L3/H3 after breaking H4/L4
        # Long: price pulled back to L3 after being above H4 (bullish continuation)
        long_setup = (close[i-1] > H4_aligned[i-1]) and (low[i] <= L3_aligned[i])
        # Short: price pulled back to H3 after being below L4 (bearish continuation)
        short_setup = (close[i-1] < L4_aligned[i-1]) and (high[i] >= H3_aligned[i])
        
        # Entry with volume expansion and filters
        long_entry = long_setup and volume_spike[i] and uptrend and vol_filter
        short_entry = short_setup and volume_spike[i] and downtrend and vol_filter
        
        # Stop loss: H4 for longs, L4 for shorts
        stop_long = position == 1 and high[i] >= H4_aligned[i]
        stop_short = position == -1 and low[i] <= L4_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif stop_long or stop_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_1d_Camarilla_Pullback_Strategy"
timeframe = "4h"
leverage = 1.0