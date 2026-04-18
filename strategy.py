#!/usr/bin/env python3
"""
1h_4D1D_Combined_Breakout_Volume_Trend
Hypothesis: Combine 4h and 1d timeframe breakouts for institutional momentum. 
Long when price breaks above 4h high AND 1d high with volume confirmation and 1d uptrend.
Short when price breaks below 4h low AND 1d low with volume confirmation and 1d downtrend.
Uses 1h timeframe for precise entry timing, 4h/1d for signal direction to reduce trade frequency.
Session filter (08-20 UTC) avoids low-liquidity periods. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for structure
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d data for trend and higher timeframe structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h structure: recent high/low for breakout
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1d structure: daily high/low for breakout confirmation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 4h structure to 1h
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    
    # Align 1d structure to 1h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume filter: >1.8x 20-period average to avoid noise
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # 1d EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(high_4h_aligned[i]) or np.isnan(low_4h_aligned[i]) or
            np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_4h_val = high_4h_aligned[i]
        low_4h_val = low_4h_aligned[i]
        high_1d_val = high_1d_aligned[i]
        low_1d_val = low_1d_aligned[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_aligned[i]
        
        if position == 0:
            # Long: break above both 4h high AND 1d high with volume in uptrend
            if price > high_4h_val and price > high_1d_val and vol_ok and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: break below both 4h low AND 1d low with volume in downtrend
            elif price < low_4h_val and price < low_1d_val and vol_ok and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position: hold until breakdown below 4h low OR trend reversal
            if price < low_4h_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short position: hold until breakout above 4h high OR trend reversal
            if price > high_4h_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4D1D_Combined_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0