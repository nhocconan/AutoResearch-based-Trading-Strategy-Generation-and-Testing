#!/usr/bin/env python3
"""
1d_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeConfirm
Hypothesis: Daily Camarilla H4/L4 breakout with weekly EMA50 trend filter and volume confirmation.
Designed for 7-25 trades/year (30-100 over 4 years) to minimize fee drag.
Uses tight entry conditions: breakout + volume spike + 1w EMA50 trend alignment.
Works in bull markets via breakout continuation and bear markets via trend following.
ATR-based stoploss for risk management.
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
    
    # 1w data for EMA50 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 trend filter (loaded ONCE)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w data for Camarilla calculation (loaded ONCE)
    df_1w_camarilla = get_htf_data(prices, '1w')
    
    # Prior 1w bar OHLC for Camarilla calculation
    prev_close = df_1w_camarilla['close'].shift(1).values
    prev_high = df_1w_camarilla['high'].shift(1).values
    prev_low = df_1w_camarilla['low'].shift(1).values
    
    # Camarilla levels: H4, L4 (strong breakout levels)
    camarilla_range = prev_high - prev_low
    h4 = prev_close + camarilla_range * 1.1 / 2
    l4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (completed 1w bar)
    h4_aligned = align_htf_to_ltf(prices, df_1w_camarilla, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w_camarilla, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss calculation
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1w EMA (50), volume MA (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H4/L4 breakout + volume spike + 1w EMA50 trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be on correct side of 1w EMA50
            long_trend = curr_close > ema_50_1w_aligned[i]
            short_trend = curr_close < ema_50_1w_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla H4 (failed breakout) 
            # or trend reverses or ATR stoploss hit
            atr_stop = entry_price - 2.5 * atr[i]
            if curr_close < h4_aligned[i] or curr_close < ema_50_1w_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla L4 (failed breakout) 
            # or trend reverses or ATR stoploss hit
            atr_stop = entry_price + 2.5 * atr[i]
            if curr_close > l4_aligned[i] or curr_close > ema_50_1w_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0