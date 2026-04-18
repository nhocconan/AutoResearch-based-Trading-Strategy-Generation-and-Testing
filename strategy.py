# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_1dTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla H3/L3 levels act as strong support/resistance. Breakouts with volume spike and daily EMA(34) trend filter capture momentum in both bull and bear markets. Target: 15-25 trades/year on 12h timeframe.
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
    
    # Get daily data for Camarilla calculation and EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (H3, L3) using standard formula
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    range_1d = high_1d - low_1d
    # Camarilla H3 = close + (high - low) * 1.1/4
    # Camarilla L3 = close - (high - low) * 1.1/4
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    
    # Shift by 1 to use previous day's levels only (no look-ahead)
    h3_prev = h3.shift(1).values
    l3_prev = l3.shift(1).values
    
    # Align to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    
    # Get daily data for EMA trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for volatility filter (14-period on 12h)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 3.0x 20-period average on 12h (tighter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above H3 with volume spike, price above daily EMA, and sufficient volatility
            if price > h3_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume spike, price below daily EMA, and sufficient volatility
            elif price < l3_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 2 bars (24 hours for 12h)
            if bars_since_entry < 2:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to L3 or breaks below daily EMA
                if price <= l3_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 2 bars (24 hours for 12h)
            if bars_since_entry < 2:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to H3 or breaks above daily EMA
                if price >= h3_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Camarilla_H3L3_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0