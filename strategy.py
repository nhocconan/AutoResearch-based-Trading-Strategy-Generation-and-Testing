#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    
    # 12h ATR for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr_12h = np.maximum(high_12h - low_12h, np.absolute(high_12h - np.roll(close_12h, 1)), np.absolute(low_12h - np.roll(close_12h, 1)))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # 12h close for trend filter (EMA34)
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 12h Camarilla pivot levels (based on previous day)
    # Camarilla formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    camarilla_mult = 1.1 / 12
    r1_12h = prev_close_12h + (prev_high_12h - prev_low_12h) * camarilla_mult
    s1_12h = prev_close_12h - (prev_high_12h - prev_low_12h) * camarilla_mult
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # 4h volume spike detection (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(atr_12h_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or \
           np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_spike = volume[i] > vol_ma20[i] * 1.5
        
        # Trend bias from 12h EMA34
        long_bias = price > ema34_12h_aligned[i]
        short_bias = price < ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price crosses above R1 with volume spike and long bias
            if price > r1_12h_aligned[i] and vol_spike and long_bias:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S1 with volume spike and short bias
            elif price < s1_12h_aligned[i] and vol_spike and short_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below S1 or ATR-based stop
            if price < s1_12h_aligned[i] or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above R1 or ATR-based stop
            if price > r1_12h_aligned[i] or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals