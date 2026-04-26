#!/usr/bin/env python3
"""
6h_ADX_Alligator_Combo
Hypothesis: Combines ADX(14) for trend strength with Williams Alligator (SMAs 13/8/5, 8/5/3) on 6h to filter whipsaws. Long when ADX>25 + price > Alligator Jaw (13) + Teeth (8) > Lips (5). Short when ADX>25 + price < Jaw + Teeth < Lips. Uses discrete sizing (0.25) and ATR(14) stoploss (2.0x). Designed for 6h to capture medium trends with low trade frequency (target 12-37/year) while avoiding ranging markets. Works in bull/bear by requiring strong trend confirmation.
"""

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
    
    # Calculate ADX(14) on 6h
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_raw = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_raw
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_raw
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams Alligator (SMAs 13/8/5, 8/5/3) on 6h
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of ADX(14), Alligator jaws/teeth/lips, ATR
    start_idx = max(14+14, 13, 8, 5, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: strong trend + price above jaw + teeth above lips
            long_signal = strong_trend and (close_val > jaw[i]) and (teeth[i] > lips[i])
            
            # Short: strong trend + price below jaw + teeth below lips
            short_signal = strong_trend and (close_val < jaw[i]) and (teeth[i] < lips[i])
            
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
            # Exit: trend weakens OR price hits ATR stoploss
            if (adx[i] <= 25) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend weakens OR price hits ATR stoploss
            if (adx[i] <= 25) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_Combo"
timeframe = "6h"
leverage = 1.0