#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_With_Volume_and_1wTrend
Hypothesis: Breakouts above Keltner upper band with volume confirmation and above 1-week EMA50 indicate strong momentum in bull markets, while breakdowns below lower band with volume and below 1-week EMA50 capture bear market moves. Keltner channels adapt to volatility better than fixed bands, and weekly trend ensures alignment with higher-timeframe momentum. Designed for low trade frequency (target 20-50/year) to minimize fee decay while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0) on 4h
    # Middle = EMA20, Width = ATR(10) * 2
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(high - low).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper = ema_20 + 2.0 * atr_10
    lower = ema_20 - 2.0 * atr_10
    
    # Volume spike: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1-week EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        up = upper[i]
        lowb = lower[i]
        vol_spike = volume_spike[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price > upper band with volume spike and above weekly EMA50
            if price > up and vol_spike and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: price < lower band with volume spike and below weekly EMA50
            elif price < lowb and vol_spike and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < middle line (EMA20) or loss of weekly trend
            if price < ema_20[i] or price < ema_50_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > middle line or above weekly EMA50
            if price > ema_20[i] or price > ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Channel_Breakout_With_Volume_and_1wTrend"
timeframe = "4h"
leverage = 1.0