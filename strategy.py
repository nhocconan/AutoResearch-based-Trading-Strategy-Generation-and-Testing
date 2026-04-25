#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) act as strong intraday support/resistance. 
Breakouts above R1 or below S1 with 1d EMA34 trend alignment and volume confirmation 
capture high-probability momentum moves. Works in both bull/bear via trend filter. 
Target: 20-40 trades/year on 4h (80-160 total over 4 years).
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
    open_prices = prices['open'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Camarilla pivot levels from previous day (using typical price)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # Calculate daily typical price, high, low, close from 1d data
    df_1d_typical = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    typical_1d = df_1d_typical.values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rang = high_1d - low_1d
    R1 = close_1d + (1.1 * rang / 12)
    S1 = close_1d - (1.1 * rang / 12)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 1) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i])):
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
            # Look for entry signals - require: Camarilla breakout + trend + volume
            # Long: price breaks above R1 AND bullish bias AND volume spike
            long_entry = (curr_high > R1_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below S1 AND bearish bias AND volume spike
            short_entry = (curr_low < S1_aligned[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below S1 (reversal) OR loss of bullish bias
            if (curr_low < S1_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above R1 (reversal) OR loss of bearish bias
            if (curr_high > R1_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0