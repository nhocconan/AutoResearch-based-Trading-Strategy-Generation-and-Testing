#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation (>2x 20-bar avg). Enters long when price breaks above H3 in 1w uptrend, short when breaks below L3 in 1w downtrend. Uses ATR-based stoploss and discrete sizing (0.25) to limit fee churn. Designed for 12h timeframe with ~12-37 trades/year, works in bull/bear by following 1w trend filter.
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
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14 periods)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
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
    
    # Start index: need 20-period data for volume MA and 50 for 1w EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Calculate Camarilla levels for current 12h bar using previous week's OHLC
        # Get the previous completed 1w bar
        if i < len(df_1w):
            prev_close_1w = close_1w[i-1] if i-1 >= 0 else close_1w[0]
            prev_high_1w = high_1w[i-1] if i-1 >= 0 else high_1w[0]
            prev_low_1w = low_1w[i-1] if i-1 >= 0 else low_1w[0]
        else:
            prev_close_1w = close_1w[-1]
            prev_high_1w = high_1w[-1]
            prev_low_1w = low_1w[-1]
        
        # Camarilla levels calculation (H3/L3)
        range_1w = prev_high_1w - prev_low_1w
        camarilla_h3 = prev_close_1w + (range_1w * 1.1 / 4)
        camarilla_l3 = prev_close_1w - (range_1w * 1.1 / 4)
        
        # Align Camarilla levels to 12h timeframe (they change only when 1w bar changes)
        camarilla_h3_aligned = camarilla_h3
        camarilla_l3_aligned = camarilla_l3
        
        if position == 0:
            # Long: price breaks above H3 in 1w uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_h3_aligned) and \
                              (close_1w[i] > ema_50_1w_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below L3 in 1w downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_l3_aligned) and \
                              (close_1w[i] < ema_50_1w_aligned[i]) and \
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
            if (curr_close < camarilla_l3_aligned) or \
               (curr_close < atr_stop) or \
               (close_1w[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR stoploss hit OR trend turns up
            if (curr_close > camarilla_h3_aligned) or \
               (curr_close > atr_stop) or \
               (close_1w[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0