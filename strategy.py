#!/usr/bin/env python3
"""
1h_4H_1D_Trend_Filter_With_Volume_Spike_Entry
Hypothesis: Use 4h EMA20 and 1d EMA50 as trend filters, enter on 1h breakouts of 20-period high/low with volume spike confirmation. Trend filters reduce false breakouts in choppy markets. Volume spike ensures momentum confirmation. Works in bull (long when above both EMAs) and bear (short when below both EMAs). Target: 60-120 total trades over 4 years = 15-30/year.
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
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for stoploss and volatility filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 20-period high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA20 (20), EMA50 (50), ATR (14), volume MA (20), breakout (20)
    start_idx = max(20, 50, 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_20_val = ema_20_4h_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        
        if position == 0:
            # Long: Break above 20-period high, above both EMAs, with volume spike
            long_signal = (high_val > high_20_val) and (close_val > ema_20_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: Break below 20-period low, below both EMAs, with volume spike
            short_signal = (low_val < low_20_val) and (close_val < ema_20_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Close below either EMA (trend change) OR trailing stop (2.0*ATR below high)
            if (close_val < ema_20_val) or (close_val < ema_50_val) or (close_val < highest_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Close above either EMA (trend change) OR trailing stop (2.0*ATR above low)
            if (close_val > ema_20_val) or (close_val > ema_50_val) or (close_val > lowest_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4H_1D_Trend_Filter_With_Volume_Spike_Entry"
timeframe = "1h"
leverage = 1.0