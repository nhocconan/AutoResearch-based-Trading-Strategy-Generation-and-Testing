#!/usr/bin/env python3
"""
6h Camarilla H3L3 Breakout + 12h EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance. Breakouts beyond these levels with volume confirmation
and 12h EMA34 trend filter capture institutional flow. Works in bull/bear via trend filter. Target: 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h: H3, L3, H4, L4
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Use previous completed 12h bar for level calculation (avoid look-ahead)
    # We'll calculate levels on completed bar and align
    rng = high_12h - low_12h
    H4 = close_12h + 1.1 * rng * 1.1 / 2
    H3 = close_12h + 1.1 * rng * 1.1 / 4
    L3 = close_12h - 1.1 * rng * 1.1 / 4
    L4 = close_12h - 1.1 * rng * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (no extra delay needed for pivot points)
    H4_aligned = align_htf_to_ltf(prices, df_12h, H4)
    H3_aligned = align_htf_to_ltf(prices, df_12h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_12h, L3)
    L4_aligned = align_htf_to_ltf(prices, df_12h, L4)
    
    # Get 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        H4_val = H4_aligned[i]
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        L4_val = L4_aligned[i]
        ema_34 = ema_34_12h_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average (stricter for 6h)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        if position == 0:
            # Long: price breaks above H3 (resistance) AND volume spike AND uptrend
            long_condition = (curr_high > H3_val) and volume_spike and uptrend
            # Short: price breaks below L3 (support) AND volume spike AND downtrend
            short_condition = (curr_low < L3_val) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or trend reversal
            if curr_close <= entry_price - 2.5 * atr_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or trend reversal
            if curr_close >= entry_price + 2.5 * atr_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0