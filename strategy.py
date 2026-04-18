#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_Volume_Trend
Hypothesis: Price breaking out of Keltner Channel (EMA20 ± ATR*2) with volume spike and ADX>25 indicates strong momentum.
Exit when price returns inside the Keltner Channel or ADX weakens (<20). Designed for low trade frequency to avoid fee drag
while capturing strong trending moves in both bull and bear markets. Uses Keltner instead of Bollinger Bands for better
trend-following properties in volatile crypto markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel: EMA20(close) ± ATR(10)*2
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    kc_upper = ema20 + 2.0 * atr10
    kc_lower = ema20 - 2.0 * atr10
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ADX(14) trend strength filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr_dm = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    tr_dm = np.concatenate([[0], tr_dm])
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr_dm).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 10, 14*2)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(volume_spike[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = kc_upper[i]
        lower = kc_lower[i]
        vol_spike = volume_spike[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner band with volume spike and strong trend
            if price > upper and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band with volume spike and strong trend
            elif price < lower and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns inside Keltner Channel OR ADX weakens
            if price < upper and price > lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns inside Keltner Channel OR ADX weakens
            if price < upper and price > lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Channel_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0