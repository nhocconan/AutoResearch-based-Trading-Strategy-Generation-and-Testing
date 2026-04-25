#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20-bar avg). Enters long when price breaks above H3 in 1d uptrend, short when breaks below L3 in 1d downtrend. Uses ATR(14) stoploss and discrete sizing (0.25) to limit fee churn. Designed for 12h timeframe with ~12-37 trades/year, works in bull/bear by following 1d trend filter.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14 periods)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need 20-period data for volume MA and 50 for 1d EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Get previous completed 1d bar for Camarilla calculation
        # We use index i-1 for 1d data since current 1d bar is not complete until 12h bar closes
        idx_1d = i // 2  # 2x 12h bars per 1d (since 12h * 2 = 24h = 1d)
        if idx_1d < 1:
            # Not enough 1d history yet
            signals[i] = 0.0
            continue
            
        prev_close_1d = close_1d[idx_1d - 1]
        prev_high_1d = high_1d[idx_1d - 1]
        prev_low_1d = low_1d[idx_1d - 1]
        
        # Camarilla levels calculation (H3/L3)
        range_1d = prev_high_1d - prev_low_1d
        camarilla_h3 = prev_close_1d + (range_1d * 1.1 / 4)
        camarilla_l3 = prev_close_1d - (range_1d * 1.1 / 4)
        
        if position == 0:
            # Long: price breaks above H3 in 1d uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_h3) and \
                              (close_1d[idx_1d] > ema_50_1d_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below L3 in 1d downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_l3) and \
                              (close_1d[idx_1d] < ema_50_1d_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (2.0 * atr[i])
            elif bearish_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below L3 OR stoploss hit OR trend turns down
            if (curr_close < camarilla_l3) or \
               (curr_close < atr_stop) or \
               (close_1d[idx_1d] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR stoploss hit OR trend turns up
            if (curr_close > camarilla_h3) or \
               (curr_close > atr_stop) or \
               (close_1d[idx_1d] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0