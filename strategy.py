#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Volume_Trend_Filter_v1
Hypothesis: Camarilla pivot (R4/S4) breakout with 1d EMA50 trend filter and volume spike confirmation on 4h timeframe.
Long when price breaks above R4 with volume > 2.0x average and uptrend (close > EMA50).
Short when price breaks below S4 with volume > 2.0x average and downtrend (close < EMA50).
Uses discrete sizing 0.25 to minimize fee churn. ATR stoploss exits when price moves 3.0x ATR against position.
Designed to work in both bull and bear markets by following the daily trend while using extreme Camarilla levels for high-probability entries.
Target trades: 15-30/year (60-120 total over 4 years) to stay well below fee drag threshold.
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
    
    # Get 1d data for Camarilla levels and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R4 and S4 levels (extreme breakout levels)
    R4 = close_1d_prev + (high_1d - low_1d) * 1.1 / 2
    S4 = close_1d_prev - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels (no additional delay needed as they're based on completed 1d bar)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: 2.0x average volume (stricter filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (50), volume MA (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        R4_val = R4_aligned[i]
        S4_val = S4_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume confirmation and uptrend
            long_signal = (high_val > R4_val) and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_50_1d_val)
            # Short: price breaks below S4 with volume confirmation and downtrend
            short_signal = (low_val < S4_val) and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR stoploss or trend reversal
            if close_val < entry_price - 3.0 * atr_val or close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR stoploss or trend reversal
            if close_val > entry_price + 3.0 * atr_val or close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0