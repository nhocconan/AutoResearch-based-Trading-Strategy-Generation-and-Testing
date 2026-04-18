#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_1wEMA34_VolumeSpike
Hypothesis: Weekly H3/L3 Camarilla levels on 1d timeframe act as strong support/resistance. 
Breakouts with volume spike and weekly EMA(34) trend filter capture momentum in both bull and bear markets.
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
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
    
    # Get daily data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels using standard formula
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    range_1d = high_1d - low_1d
    close_prev = close_1d.shift(1)
    
    # Camarilla levels based on previous day
    H3 = close_prev + (range_1d * 1.1 / 6)
    L3 = close_prev - (range_1d * 1.1 / 6)
    H4 = close_prev + (range_1d * 1.1 / 2)
    L4 = close_prev - (range_1d * 1.1 / 2)
    
    # Shift by 1 to use previous day's levels only
    H3_prev = H3.shift(1).values
    L3_prev = L3.shift(1).values
    H4_prev = H4.shift(1).values
    L4_prev = L4.shift(1).values
    
    # Align to 1d timeframe (already aligned since we're using daily data)
    H3_aligned = H3_prev  # Already aligned to daily
    L3_aligned = L3_prev
    H4_aligned = H4_prev
    L4_aligned = L4_prev
    
    # Get weekly data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close']
    
    # Calculate weekly EMA(34)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # ATR for volatility filter (14-period on 1d)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 2.0x 20-period average on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 50  # enough for EMA34 and other indicators
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        H4_val = H4_aligned[i]
        L4_val = L4_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above H3 with volume spike, price above weekly EMA, and sufficient volatility
            if price > H3_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume spike, price below weekly EMA, and sufficient volatility
            elif price < L3_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to L3 or breaks below weekly EMA
                if price <= L3_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to H3 or breaks above weekly EMA
                if price >= H3_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "1d_Camarilla_H3L3_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0