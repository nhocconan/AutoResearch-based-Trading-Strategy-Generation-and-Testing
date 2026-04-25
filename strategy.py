#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Uses 1d for signal direction to minimize overtrading, 12h for precise entry timing.
Volume spike confirms momentum. Designed for 12-37 trades/year on BTC/ETH, works in both bull and bear
by following the higher timeframe trend. Uses ATR-based stoploss for risk management.
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
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + 1.1 * prev_range * (1.0/12.0)  # R1 = C + 1.1*(HL/12)
    S1 = prev_close - 1.1 * prev_range * (1.0/12.0)  # S1 = C - 1.1*(HL/12)
    
    # Align 1d levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (12h ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and indicators
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume spike and trend alignment
            # Long breakout: price breaks above R1 with uptrend and volume spike
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i]
            # Short breakout: price breaks below S1 with downtrend and volume spike
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i]
            
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
            # Stoploss: 2.0 * ATR below entry
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks below S1 (mean reversion) or trend changes
            elif curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Stoploss: 2.0 * ATR above entry
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes
            elif curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "12h"
leverage = 1.0