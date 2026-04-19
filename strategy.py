#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with daily VWAP deviation for mean reversion.
# Long when: CHOP(14) > 61.8 (ranging market) AND price < VWAP - 1.5*ATR(20) (oversold)
# Short when: CHOP(14) > 61.8 (ranging market) AND price > VWAP + 1.5*ATR(20) (overbought)
# Exit when: Price crosses back through VWAP (mean reversion complete)
# Choppiness Index identifies ranging markets ideal for mean reversion, VWAP acts as dynamic fair value,
# ATR-scaled deviations prevent entries in low volatility. Works in both bull (buy dips) and bear (sell rallies).
name = "4h_Choppiness_VWAP_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (Volume Weighted Average Price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = vwap_num / vwap_den
    
    # Calculate ATR(20) for volatility scaling
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    atr[atr_period] = np.mean(tr[:atr_period])
    for i in range(atr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Choppiness Index (14) for regime detection
    chop_period = 14
    atr_sum = np.zeros_like(close)
    atr_sum[chop_period] = np.sum(tr[:chop_period])
    for i in range(chop_period + 1, len(tr)):
        atr_sum[i] = atr_sum[i-1] - tr[i-chop_period] + tr[i]
    
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    max_high[chop_period] = np.max(high[:chop_period+1])
    min_low[chop_period] = np.min(low[:chop_period+1])
    for i in range(chop_period + 1, len(high)):
        max_high[i] = max(max_high[i-1], high[i])
        min_low[i] = min(min_low[i-1], low[i])
    
    chop = np.full_like(close, 50.0)  # Default to neutral
    valid_range = (max_high - min_low) > 0
    chop[chop_period:] = 100 * np.log10(atr_sum[chop_period:] / (max_high[chop_period:] - min_low[chop_period:])) / np.log10(chop_period)
    
    # Align daily data to 4H timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, chop_period) + 1  # Wait for ATR and CHOP calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_aligned[i]
        
        if position == 0:
            # Long entry: Choppy market (range) AND price significantly below VWAP (oversold)
            if (chop_val > 61.8 and price < vwap - 1.5 * atr_val):
                signals[i] = 0.25
                position = 1
            # Short entry: Choppy market (range) AND price significantly above VWAP (overbought)
            elif (chop_val > 61.8 and price > vwap + 1.5 * atr_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back above VWAP (mean reversion)
            if price > vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back below VWAP (mean reversion)
            if price < vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals