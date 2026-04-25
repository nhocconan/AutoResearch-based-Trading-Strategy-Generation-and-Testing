#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter and volume confirmation (>2x 20-bar avg). Enters long when price breaks above R1 in weekly uptrend, short when breaks below S1 in weekly downtrend. Uses ATR-based stoploss and discrete sizing (0.30) to limit fee churn. Designed for 1d timeframe with ~7-25 trades/year, works in bull/bear by following weekly trend filter.
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
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
    
    # Start index: need 20-period data for volume MA and 34 for 1w EMA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Get previous completed 1d bar for Camarilla calculation
        prev_idx_1d = i - 1
        if prev_idx_1d < 0:
            signals[i] = 0.0
            continue
            
        prev_close = close[prev_idx_1d]
        prev_high = high[prev_idx_1d]
        prev_low = low[prev_idx_1d]
        
        # Camarilla levels calculation
        range_1d = prev_high - prev_low
        if range_1d <= 0:  # Avoid division by zero or invalid range
            signals[i] = 0.0
            continue
            
        camarilla_r1 = prev_close + (range_1d * 1.1 / 12)
        camarilla_s1 = prev_close - (range_1d * 1.1 / 12)
        
        if position == 0:
            # Long: price breaks above R1 in weekly uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_r1) and \
                              (close[i] > ema_34_1w_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below S1 in weekly downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_s1) and \
                              (close[i] < ema_34_1w_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (1.5 * atr[i])
            elif bearish_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (1.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price breaks below S1 OR stoploss hit OR trend turns down
            if (curr_close < camarilla_s1) or \
               (curr_close < atr_stop) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price breaks above R1 OR stoploss hit OR trend turns up
            if (curr_close > camarilla_r1) or \
               (curr_close > atr_stop) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0