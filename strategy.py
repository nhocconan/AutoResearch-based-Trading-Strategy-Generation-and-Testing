#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Uses 1d for trend to avoid whipsaw and reduce overtrading. Targets 12-37 trades/year on BTC/ETH
by requiring confluence of breakout, trend, and volume. Designed to work in both bull and bear
markets via strict entry conditions and ATR-based stoploss.
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
    
    # 1d data for EMA34 trend filter and Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + 1.1 * prev_range * (1.0/12)  # R1 = C + 1.1*(HL/12)
    S1 = prev_close - 1.1 * prev_range * (1.0/12)  # S1 = C - 1.1*(HL/12)
    
    # Align 1d levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and 1d indicators
    start_idx = 34
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long breakout: price breaks above R1 with uptrend and volume confirmation
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below S1 with downtrend and volume confirmation
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_confirm[i]
            
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
            # Stoploss: 2.5 * ATR below entry (using 12h ATR approximation)
            # Calculate 12h ATR using 12-bar period (since 12h TF)
            if i >= 12:
                tr1 = high[1:i+1] - low[1:i+1]
                tr2 = np.abs(high[1:i+1] - close[:i])
                tr3 = np.abs(low[1:i+1] - close[:i])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr[-12:]) if len(tr) >= 12 else np.nan
            else:
                atr = np.nan
            
            if not np.isnan(atr) and curr_close < entry_price - 2.5 * atr:
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
            # Calculate 12h ATR
            if i >= 12:
                tr1 = high[1:i+1] - low[1:i+1]
                tr2 = np.abs(high[1:i+1] - close[:i])
                tr3 = np.abs(low[1:i+1] - close[:i])
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = np.mean(tr[-12:]) if len(tr) >= 12 else np.nan
            else:
                atr = np.nan
            
            if not np.isnan(atr) and curr_close > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks above R1 (mean reversion) or trend changes
            elif curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0