#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance.
Breakouts above H3 or below L3 with 1d EMA34 trend alignment and volume confirmation
capture momentum moves while avoiding false breakouts. Works in bull/bear via HTF trend.
Target: 20-40 trades/year on 4h to avoid fee drag.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla H3/L3 from previous 1d bar
    # H3 = close + (high - low) * 1.1 / 6
    # L3 = close - (high - low) * 1.1 / 6
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    h3 = df_1d['close'] + range_1d * 1.1 / 6
    l3 = df_1d['close'] - range_1d * 1.1 / 6
    
    # Align H3/L3 levels to LTF
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: breakout + trend + volume
            # Long: break above H3 AND bullish bias AND volume spike
            long_entry = (curr_high > h3_aligned[i]) and bullish_bias and vol_spike
            # Short: break below L3 AND bearish bias AND volume spike
            short_entry = (curr_low < l3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below EMA34 (trend change) OR re-enters Camarilla H3/L3 range
            if (curr_close < ema_1d_aligned[i]) or (curr_low < h3_aligned[i] and curr_high > l3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above EMA34 (trend change) OR re-enters Camarilla H3/L3 range
            if (curr_close > ema_1d_aligned[i]) or (curr_low < h3_aligned[i] and curr_high > l3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0