#!/usr/bin/env python3
"""
4h_1d_4h_1w_EMA_Cross_Trend_Filter
Hypothesis: On 4h timeframe, use 1-day EMA (50) for primary trend, 1-week EMA (20) for regime filter, and 4h EMA (20) for entry timing. 
Long when: 4h EMA > 1d EMA AND 1d EMA > 1w EMA (bullish alignment). 
Short when: 4h EMA < 1d EMA AND 1d EMA < 1w EMA (bearish alignment).
Exit when alignment breaks. Uses volume confirmation (>1.5x 20-period average) to avoid false breakouts.
Designed for low turnover (target: 20-40 trades/year) to minimize fee drag while capturing sustained trends.
Works in bull/bear by following multi-timeframe EMA alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day and 1-week data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMAs
    # 1-day EMA(50) for primary trend
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 1-week EMA(20) for regime filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 4h EMA(20) for entry timing
    ema_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF EMAs to 4h timeframe
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_4h[i]) or np.isnan(ema_1w_4h[i]) or
            np.isnan(ema_4h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_4h_val = ema_4h[i]
        ema_1d_val = ema_1d_4h[i]
        ema_1w_val = ema_1w_4h[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: bullish alignment (4h > 1d > 1w) with volume
            if ema_4h_val > ema_1d_val and ema_1d_val > ema_1w_val and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment (4h < 1d < 1w) with volume
            elif ema_4h_val < ema_1d_val and ema_1d_val < ema_1w_val and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long while bullish alignment holds
            if ema_4h_val > ema_1d_val and ema_1d_val > ema_1w_val:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Maintain short while bearish alignment holds
            if ema_4h_val < ema_1d_val and ema_1d_val < ema_1w_val:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_1d_4h_1w_EMA_Cross_Trend_Filter"
timeframe = "4h"
leverage = 1.0