#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout + Volume Spike + Session Filter
Hypothesis: Camarilla pivot levels (H3/L3) act as intraday support/resistance.
Breakouts above H3 or below L3 with volume confirmation indicate institutional participation.
Session filter (08-20 UTC) focuses on London/NY overlap for higher quality moves.
Uses 4h EMA34 for trend filter to avoid counter-trend breakouts.
Target: 15-30 trades/year on 1h timeframe.
Works in bull/bear via 4h trend filter - only take breakouts in trend direction.
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
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous day
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # H4 = close + 1.1*(high-low)
    # L4 = close - 1.1*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use only previous day's data
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align pivots to 1h timeframe (previous day's levels available all day)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08:00-20:00 UTC (London/NY overlap)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = max(35, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price above/below 4h EMA34
        uptrend = curr_close > ema_34_4h_aligned[i]
        downtrend = curr_close < ema_34_4h_aligned[i]
        
        if position == 0:
            # Look for breakout entries
            # Long: break above H3 with volume spike in uptrend
            long_breakout = (curr_high > h3_aligned[i]) and vol_spike and uptrend
            # Short: break below L3 with volume spike in downtrend
            short_breakout = (curr_low < l3_aligned[i]) and vol_spike and downtrend
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: break below L3 (failed breakout) or reverse signal
            if curr_low < l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: break above H3 (failed breakdown) or reverse signal
            if curr_high > h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_VolumeSpike_Session_EMA34"
timeframe = "1h"
leverage = 1.0