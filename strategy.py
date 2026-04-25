#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R1 level in an uptrend (1d EMA34) with volume confirmation (>1.5x 20-bar average), sell when price breaks below S1 level in a downtrend. Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (target: 20-50 trades/year) and works in both bull and bear markets by requiring trend alignment.
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
    
    # 1d data for Camarilla pivots and EMA trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    # R1 = close + 1.125*(high-low), S1 = close - 1.125*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    R1 = prev_close + 1.125 * prev_range
    S1 = prev_close - 1.125 * prev_range
    
    # Align 1d pivot levels to 4h timeframe (waits for completed 1d bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for 1d indicators (34 for EMA, 20 for vol MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend determination based on 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 in uptrend with volume spike
            long_entry = (curr_close > R1_aligned[i]) and uptrend and volume_spike[i]
            # Short: price breaks below S1 in downtrend with volume spike
            short_entry = (curr_close < S1_aligned[i]) and downtrend and volume_spike[i]
            
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
            # Long position: exit when price breaks below S1 (trend reversal)
            if curr_close < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above R1 (trend reversal)
            if curr_close > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0