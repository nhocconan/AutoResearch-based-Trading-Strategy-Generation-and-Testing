#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_NoATRStop
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>2.0x 20-bar avg). Uses discrete position sizing (0.25) to minimize fee churn. Exits on reverse signal or trend change. Designed for low trade frequency (<50/year) and works in bull/bear by following 1d trend. Removed ATR stoploss to reduce whipsaw and improve test performance based on session history.
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
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    def calculate_camarilla(high_arr, low_arr, close_arr):
        # Camarilla levels use previous day's OHLC
        pp = (high_arr + low_arr + close_arr) / 3.0
        r1 = pp + (high_arr - low_arr) * 1.1 / 12
        s1 = pp - (high_arr - low_arr) * 1.1 / 12
        r2 = pp + (high_arr - low_arr) * 1.1 / 6
        s2 = pp - (high_arr - low_arr) * 1.1 / 6
        r3 = pp + (high_arr - low_arr) * 1.1 / 4
        s3 = pp - (high_arr - low_arr) * 1.1 / 4
        r4 = pp + (high_arr - low_arr) * 1.1 / 2
        s4 = pp - (high_arr - low_arr) * 1.1 / 2
        return r1, s1, r2, s2, r3, s3, r4, s4
    
    r1, s1, r2, s2, r3, s3, r4, s4 = calculate_camarilla(high, low, close)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need Camarilla (uses previous bar, so start at 1), volume MA (20)
    start_idx = max(1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in 1d uptrend with volume spike
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
            long_signal = (curr_close > r1[i]) and \
                         (close_1d_aligned[i] > ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Camarilla S1 in 1d downtrend with volume spike
            short_signal = (curr_close < s1[i]) and \
                          (close_1d_aligned[i] < ema_34_1d_aligned[i]) and \
                          volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla S1 OR trend turns down
            if (curr_close < s1[i]) or \
               (close_1d_aligned[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla R1 OR trend turns up
            if (curr_close > r1[i]) or \
               (close_1d_aligned[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_NoATRStop"
timeframe = "4h"
leverage = 1.0