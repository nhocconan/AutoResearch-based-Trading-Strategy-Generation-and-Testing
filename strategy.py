#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d ATR volatility filter and volume confirmation
# Uses 1d Camarilla H3/L3 levels as breakout triggers, filtered by 1d ATR expansion (>1.2x 20-period average)
# Volume confirmation requires current volume > 1.5x 20-period average to avoid false breakouts
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: ATR filter ensures we only trade during volatility expansion, avoiding ranging markets

name = "12h_1d_camarilla_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3 and L3
    h3 = pp + (range_1d * 1.1 / 4.0)
    l3 = pp - (range_1d * 1.1 / 4.0)
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 1d ATR (20-period) for volatility filter
    # TR = max(H-L, |H-PC|, |L-PC|)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR = EMA of TR with alpha = 1/period
    atr = np.full_like(tr, np.nan)
    period = 20
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])  # Simple average for first value
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    # Calculate 20-period average ATR for comparison
    atr_ma = np.full_like(atr, np.nan)
    for i in range(len(atr)):
        if i < period:
            atr_ma[i] = np.nan
        else:
            atr_ma[i] = np.mean(atr[i-period:i])
    
    # Align 1d ATR and its MA to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_12h = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Calculate 20-period average volume for volume confirmation (12h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(atr_12h[i]) or np.isnan(atr_ma_12h[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 1.2x 20-period average ATR (expanding volatility)
        volatility_expansion = atr_12h[i] > 1.2 * atr_ma_12h[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        volume_confirmation = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR volatility contracts (< 0.8x ATR MA)
            if close[i] < l3_12h[i] or atr_12h[i] < 0.8 * atr_ma_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR volatility contracts (< 0.8x ATR MA)
            if close[i] > h3_12h[i] or atr_12h[i] < 0.8 * atr_ma_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volatility expansion and volume confirmation
            if volatility_expansion and volume_confirmation:
                # Long entry: price closes above H3 (bullish breakout)
                if close[i] > h3_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price closes below L3 (bearish breakout)
                elif close[i] < l3_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals