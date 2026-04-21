#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume_ATR_v1
Hypothesis: Breakout above Camarilla H3 or below L3 on 1d with volume confirmation and ATR filter.
Works in bull/bear by capturing breakouts from key intraday levels with volatility-adjusted sizing.
Long when price breaks above H3 with volume spike and ATR expansion.
Short when price breaks below L3 with volume spike and ATR reduction.
Exit when price returns to H3/L3 or opposite pivot level.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: H3, L3, H4, L4
    rang = prev_high - prev_low
    h3 = prev_close + 1.1 * rang / 4
    l3 = prev_close - 1.1 * rang / 4
    h4 = prev_close + 1.1 * rang / 2
    l4 = prev_close - 1.1 * rang / 2
    
    # Align to 1d timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Load 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ATR for volatility filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # ATR filter: current ATR > 0.8 * 20-period average ATR (avoid low volatility)
        if i >= 20:
            atr_ma = atr[i-20:i].mean()
            atr_ok = atr[i] > 0.8 * atr_ma
        else:
            atr_ok = False
        
        if position == 0:
            # Long conditions: break above H3 with volume and ATR confirmation
            if (price > h3_aligned[i] and volume_ok and atr_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below L3 with volume and ATR confirmation
            elif (price < l3_aligned[i] and volume_ok and atr_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to H3 or reach opposite H4
            if price <= h3_aligned[i] or price >= h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to L3 or reach opposite L4
            if price >= l3_aligned[i] or price <= l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout_Volume_ATR_v1"
timeframe = "1d"
leverage = 1.0