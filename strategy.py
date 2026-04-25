#!/usr/bin/env python3
"""
1d Camarilla Pivot H3/L3 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: Weekly EMA50 defines the institutional trend. Breakouts above Camarilla H3 
or below L3 with volume spike and trend alignment capture strong momentum moves. 
In bull markets: trend-following breakouts work. In bear markets: failed breaks at 
H3/L3 trigger reversals to H4/L4 (not implemented here but logic holds via trend filter).
Uses weekly and daily HTF data loaded ONCE before loop. Target: 12-37 trades/year.
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
    open_ = prices['open'].values
    
    # ATR for stop (optional, using signal=0 for exit)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Weekly data for EMA50 trend (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Camarilla pivot points (H3, L3, H4, L4) (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Camarilla levels based on previous day's OHLC
    prev_day_close = df_1d['close'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_range = prev_day_high - prev_day_low
    camarilla_h3 = prev_day_close + (prev_day_range * 1.1 / 4)
    camarilla_l3 = prev_day_close - (prev_day_range * 1.1 / 4)
    camarilla_h4 = prev_day_close + (prev_day_range * 1.1 / 2)
    camarilla_l4 = prev_day_close - (prev_day_range * 1.1 / 2)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly EMA (50) and daily data (2 for shift)
    start_idx = max(50, 2) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > camarilla_h3_aligned[i]
        breakout_short = curr_close < camarilla_l3_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + weekly EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_1w_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_1w_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Camarilla L4 retrace or trend change
            if curr_close < camarilla_l4_aligned[i] or curr_close < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Camarilla H4 retrace or trend change
            if curr_close > camarilla_h4_aligned[i] or curr_close > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0