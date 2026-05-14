#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v4
Hypothesis: Further tighten entry conditions from v3 to reduce trade count and fee drag.
Increase volume spike threshold from 2.0x to 2.5x and increase ATR stoploss multiplier from 2.5 to 3.0.
This should reduce whipsaw and overtrading while maintaining edge in both bull and bear markets.
Target: 30-60 total trades over 4 years (7-15/year) to improve test generalization.
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
    
    # 1d data for EMA34 trend filter and Camarilla calculation (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3, L3 (strong breakout levels)
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 4
    l3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.5 * 20-period average (tighter threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.5)
    
    # ATR for stoploss calculation
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for 1d EMA (34), volume MA (20), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + 1d EMA34 trend alignment
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            # Trend filter: price must be on correct side of 1d EMA34
            long_trend = curr_close > ema_34_1d_aligned[i]
            short_trend = curr_close < ema_34_1d_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: minimum holding period + exit conditions
            if bars_since_entry < 4:  # Minimum 4 bars (16h) holding period
                signals[i] = 0.25
            else:
                # Exit when price closes below Camarilla H3 (failed breakout) 
                # or trend reverses or ATR stoploss hit
                atr_stop = entry_price - 3.0 * atr[i]  # Increased stoploss multiplier
                if curr_close < h3_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < atr_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short position: minimum holding period + exit conditions
            if bars_since_entry < 4:  # Minimum 4 bars (16h) holding period
                signals[i] = -0.25
            else:
                # Exit when price closes above Camarilla L3 (failed breakout) 
                # or trend reverses or ATR stoploss hit
                atr_stop = entry_price + 3.0 * atr[i]  # Increased stoploss multiplier
                if curr_close > l3_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > atr_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v4"
timeframe = "4h"
leverage = 1.0