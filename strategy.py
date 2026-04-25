#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v3
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>2x 20-bar avg). Enters long when price breaks above R1 in 1d uptrend, short when breaks below S1 in 1d downtrend. Uses ATR-based stoploss and discrete sizing (0.30) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 1d trend filter. Fixed alignment issues from previous versions.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Shift 1d data by 1 to get previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # Set first value to avoid roll issues
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Calculate Camarilla levels
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r1 = prev_close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = prev_close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need 20-period data for volume MA and 34 for 1d EMA
    start_idx = max(20, 34, 1)  # +1 for Camarilla calculation needing previous bar
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume confirmation
            bullish_breakout = (curr_close > camarilla_r1_aligned[i]) and \
                              (close_1d[i] > ema_34_1d_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume confirmation
            bearish_breakout = (curr_close < camarilla_s1_aligned[i]) and \
                              (close_1d[i] < ema_34_1d_aligned[i]) and \
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
            if (curr_close < camarilla_s1_aligned[i]) or \
               (curr_close < atr_stop) or \
               (close_1d[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price breaks above R1 OR stoploss hit OR trend turns up
            if (curr_close > camarilla_r1_aligned[i]) or \
               (curr_close > atr_stop) or \
               (close_1d[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v3"
timeframe = "4h"
leverage = 1.0