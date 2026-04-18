#!/usr/bin/env python3
"""
12h_StrongTrend_With_Volume_Confirmation
Hypothesis: Strong 12-hour trends (price > 50-period SMA) combined with above-average volume 
capture sustained institutional moves. Works in bull/bear by following confirmed trends.
Target: 20-40 trades/year (80-160 total over 4 years) for optimal balance.
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
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA50 trend
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Price momentum: close > open for bullish, close < open for bearish
    bullish_momentum = close > prices['open'].values
    bearish_momentum = close < prices['open'].values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(bullish_momentum[i]) or np.isnan(bearish_momentum[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_50_12h[i]
        vol_ok = volume_filter[i]
        bull_mom = bullish_momentum[i]
        bear_mom = bearish_momentum[i]
        
        if position == 0:
            # Long: price above daily EMA50 with volume and bullish momentum
            if price > ema_trend and vol_ok and bull_mom:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA50 with volume and bearish momentum
            elif price < ema_trend and vol_ok and bear_mom:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend breaks or volume dries up
            if price < ema_trend or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend breaks or volume dries up
            if price > ema_trend or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_StrongTrend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0