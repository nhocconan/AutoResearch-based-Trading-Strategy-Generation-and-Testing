#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, trade long when price breaks above Camarilla R1 level with volume spike and above 4h EMA20 trend, short when breaks below S1 with volume spike and below 4h EMA20. Uses 4h trend for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise trades. Discrete sizing (0.20) limits fee drag. Designed for 60-150 trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla levels from prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True range for prior 4h bar
    prev_close_4h = np.roll(close_4h, 1)
    prev_close_4h[0] = close_4h[0]  # first bar
    tr_4h = np.maximum(high_4h - low_4h, np.maximum(np.abs(high_4h - prev_close_4h), np.abs(low_4h - prev_close_4h)))
    atr_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Camarilla levels: based on prior bar's range
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + 1.0833 * hl_range_4h  # R1 level
    s1_4h = close_4h - 1.0833 * hl_range_4h  # S1 level
    
    # Align HTF indicators to 1h timeframe
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # ATR for stoploss calculation (1h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA20 (20), ATR (14), volume MA (20)
    start_idx = max(20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_20_val = ema_20_4h_aligned[i]
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA20, with volume spike
            long_signal = (close_val > r1_val) and (close_val > ema_20_val) and vol_spike
            
            # Short: price breaks below S1, below 4h EMA20, with volume spike
            short_signal = (close_val < s1_val) and (close_val < ema_20_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 OR ATR stoploss (2*ATR below entry)
            if (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR ATR stoploss (2*ATR above entry)
            if (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0