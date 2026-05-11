#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dATR10_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for ATR10 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) on 1d
    atr_1d = np.zeros_like(close_1d)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.abs(high_1d[1:] - close_1d[:-1]),
                    np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_1d[10:] = pd.Series(tr).rolling(window=10, min_periods=10).mean().values[10:]
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Trend: close_1d > ema34_1d + 0.5 * atr_1d (bullish) or close_1d < ema34_1d - 0.5 * atr_1d (bearish)
    bullish_trend = close_1d > (ema34_1d + 0.5 * atr_1d)
    bearish_trend = close_1d < (ema34_1d - 0.5 * atr_1d)
    
    # Align trend signals to 4h
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar
    # Using previous day's range (shifted by 1 to avoid lookahead)
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_r1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_s1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1, bullish trend, volume spike
            if (close[i] > r1_aligned[i] and 
                bullish_trend_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1, bearish trend, volume spike
            elif (close[i] < s1_aligned[i] and 
                  bearish_trend_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S1 or bearish trend
            if close[i] < s1_aligned[i] or bearish_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R1 or bullish trend
            if close[i] > r1_aligned[i] or bullish_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals