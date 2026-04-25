#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSp
Hypothesis: Daily Camarilla H3/L3 breakout with weekly EMA34 trend filter and volume spike confirmation. Uses 1d timeframe to minimize fee drag while capturing multi-day trends. Weekly EMA ensures alignment with higher timeframe momentum. Volume spike (>2.0x average) confirms institutional participation. Designed for 7-15 trades/year per symbol to avoid fee drag while maintaining edge in both bull and bear markets via trend following.
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
    
    # Load 1d data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Load 1w data ONCE for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1d Camarilla pivot levels H3/L3 (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # H3 = close + 1.5 * range (strong resistance)
    # L3 = close - 1.5 * range (strong support)
    H3 = prev_close + 1.5 * prev_range
    L3 = prev_close - 1.5 * prev_range
    
    # Align 1d pivot levels to 1d timeframe (no shift needed as we use previous day's values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d indicators (20 for vol MA) and 1w EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals: breakout with volume spike and trend alignment
            # Long breakout: price breaks above H3 with uptrend and volume spike
            long_breakout = (curr_close > H3_aligned[i]) and (curr_close > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short breakout: price breaks below L3 with downtrend and volume spike
            short_breakout = (curr_close < L3_aligned[i]) and (curr_close < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on mean reversion to midpoint or trend change
            # Exit if price breaks below L3 (mean reversion) or closes below weekly EMA
            if curr_close < L3_aligned[i] or curr_close < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on mean reversion to midpoint or trend change
            # Exit if price breaks above H3 (mean reversion) or closes above weekly EMA
            if curr_close > H3_aligned[i] or curr_close > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSp"
timeframe = "1d"
leverage = 1.0