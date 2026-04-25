#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + Volume Spike + ATR Filter (1h primary)
Hypothesis: Camarilla H3/L3 levels from 4h/1d represent strong intraday support/resistance where breakouts indicate institutional participation.
Volume confirms real money involvement, ATR filter ensures sufficient momentum. Uses 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise trades. Target: 15-37 trades/year on 1h.
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
    
    # Load 4h data ONCE before loop for HTF direction and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 4h bar (H3/L3 for breakout)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: H3 = close + (high-low)*1.1/2, L3 = close - (high-low)*1.1/2
    camarilla_h3_4h = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_l3_4h = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # 1d EMA34 for trend filter (only need completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR filter: ensure sufficient volatility (1h ATR)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > (atr_ma * 0.8)  # Trade when volatility is above 80% of its 50-period MA
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 20, 14, 50, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_filter[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        atr_ok = atr_filter[i]
        above_ema = curr_close > ema_34_aligned[i]  # Uptrend filter
        below_ema = curr_close < ema_34_aligned[i]  # Downtrend filter
        
        # Camarilla breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_long = curr_close > camarilla_h3_aligned[i-1]  # Break above previous 4h's H3
        breakout_short = curr_close < camarilla_l3_aligned[i-1]  # Break below previous 4h's L3
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume + ATR filter + session + trend
            long_entry = breakout_long and vol_spike and atr_ok and above_ema
            short_entry = breakout_short and vol_spike and atr_ok and below_ema
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches Camarilla L3 level
            if curr_close < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price retouches Camarilla H3 level
            if curr_close > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_VolumeSpike_ATRFilter_Session"
timeframe = "1h"
leverage = 1.0