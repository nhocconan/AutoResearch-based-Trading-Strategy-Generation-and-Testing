#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d volume spike and 1d trend filter (close > EMA34).
Only trade breakouts in trending 1d markets to avoid chop. Designed for lower trade frequency (~15-30/year)
to minimize fee drag while capturing strong trends in BTC/ETH. Works in bull (breakouts with trend) and 
bear (avoids false breakouts in choppy/range markets via 1d trend filter).
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
    
    # Calculate 1d Camarilla pivot levels (R1, S1) and EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 4.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 4.0
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume > (2.0 * vol_avg_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for EMA34 and volume average
    start_idx = max(100, 34, 20)  # safety buffer, EMA34, vol avg
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and 1d trend filter
            # Only trade breakouts when 1d close > EMA34 (uptrend) for longs, < EMA34 (downtrend) for shorts
            long_entry = (close_val > R1_aligned[i]) and volume_spike_aligned[i] and (close_1d_series.iloc[-1] > ema_34_1d[-1]) if hasattr(close_1d_series, 'iloc') else (close_val > R1_aligned[i]) and volume_spike_aligned[i] and (close_1d[-1] > ema_34_1d[-1])
            short_entry = (close_val < S1_aligned[i]) and volume_spike_aligned[i] and (close_1d[-1] < ema_34_1d[-1])
            
            # Fix: use current 1d close for trend check (need to get current 1d close from aligned data)
            # Since we don't have current 1d close aligned, we'll use price action as proxy: 
            # In uptrend, price tends to stay above recent lows; in downtrend below recent highs
            # Simplified: use the alignment itself - if we're getting signals, trend is embedded
            # Better approach: check if current 12h price is above/below 1d EMA (which we have aligned)
            long_entry = (close_val > R1_aligned[i]) and volume_spike_aligned[i] and (close_val > ema_34_aligned[i])
            short_entry = (close_val < S1_aligned[i]) and volume_spike_aligned[i] and (close_val < ema_34_aligned[i])
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S1 retracement or if price breaks below EMA34 (trend change)
            if close_val < S1_aligned[i] or close_val < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or if price breaks above EMA34 (trend change)
            if close_val > R1_aligned[i] or close_val > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0