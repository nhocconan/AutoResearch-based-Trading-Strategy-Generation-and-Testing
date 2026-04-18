#!/usr/bin/env python3
"""
4h_ADX_Slope_Trend_Riding
Hypothesis: Strong trends (ADX>30) with rising ADX slope confirm momentum. Enter when price crosses above/below 20-period EMA with ADX slope confirmation. Exit when ADX falls below 20 or slope turns negative. Designed for low trade frequency to avoid fee decay while capturing sustained trends in both bull and bear markets.
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
    
    # EMA(20) for trend direction
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).values
    
    # ADX(14) calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX slope (3-period slope of ADX)
    adx_series = pd.Series(adx)
    adx_slope = adx_series.diff(3).values  # 3-period change
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20, 14*2)  # Warmup for EMA20 and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(ema20[i]) or 
            np.isnan(adx[i]) or
            np.isnan(adx_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema20[i]
        adx_val = adx[i]
        slope = adx_slope[i]
        
        if position == 0:
            # Long: price above EMA20, ADX>30, and rising ADX slope
            if price > ema and adx_val > 30 and slope > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20, ADX>30, and rising ADX slope
            elif price < ema and adx_val > 30 and slope > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below EMA20 OR ADX<20 OR negative slope
            if price < ema or adx_val < 20 or slope < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above EMA20 OR ADX<20 OR negative slope
            if price > ema or adx_val < 20 or slope < 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ADX_Slope_Trend_Riding"
timeframe = "4h"
leverage = 1.0