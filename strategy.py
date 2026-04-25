#!/usr/bin/env python3
"""
6h Camarilla H3L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) from the prior 1d bar act as strong support/resistance.
Breakout above H3 or below L3 with 1d EMA34 trend alignment and volume confirmation
captures institutional momentum moves. Works in bull markets (breakouts above H3 in uptrend)
and bear markets (breakouts below L3 in downtrend). 6h timeframe targets 12-37 trades/year.
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 1d EMA alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 6h bar using prior bar's OHLC
        if i == 0:
            continue
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        rang = prev_high - prev_low
        
        # Camarilla H3 and L3 levels
        h3 = prev_close + rang * 1.1 / 2
        l3 = prev_close - rang * 1.1 / 2
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above H3 AND uptrend AND volume spike
            long_entry = (curr_close > h3) and uptrend and vol_spike
            # Short: break below L3 AND downtrend AND volume spike
            short_entry = (curr_close < l3) and downtrend and vol_spike
            
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
            # Exit: price breaks below L3 (reversal) OR loss of uptrend
            if (curr_close < l3) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above H3 (reversal) OR loss of downtrend
            if (curr_close > h3) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "6h"
leverage = 1.0