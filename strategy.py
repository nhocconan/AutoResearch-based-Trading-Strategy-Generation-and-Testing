#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSp
Hypothesis: Trade 12h timeframe with Camarilla H3/L3 breakouts filtered by weekly EMA50 trend and volume spike (2.0x 20-bar average). Uses 1d for Camarilla levels and 1w for EMA50 to reduce noise and capture strong momentum moves. Designed for low trade frequency (target 12-30/year) to minimize fee drag while maintaining edge in both bull and bear markets by requiring alignment of daily structure, weekly trend, and intraday volume confirmation.
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
    
    # 1d data for Camarilla pivot levels (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H3 = prev_close + 0.25 * prev_range  # Camarilla H3 level
    L3 = prev_close - 0.25 * prev_range  # Camarilla L3 level
    
    # Align 1d pivot levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume spike: current volume > 2.0 * 20-period average (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 1w EMA50 for trend filter (loaded ONCE)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1w EMA50 (50) and 1d indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume spike
            # Long breakout: price breaks above H3 with uptrend and volume spike
            long_breakout = (curr_close > H3_aligned[i]) and uptrend and volume_spike[i]
            # Short breakout: price breaks below L3 with downtrend and volume spike
            short_breakout = (curr_close < L3_aligned[i]) and downtrend and volume_spike[i]
            
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
            # Long position: exit conditions
            # Stoploss: 2.5 * ATR below entry
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below L3 (mean reversion) or trend changes
            elif curr_close < L3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.5 * ATR above entry
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above H3 (mean reversion) or trend changes
            elif curr_close > H3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_Trend_VolumeSp"
timeframe = "12h"
leverage = 1.0