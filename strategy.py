#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with 1w EMA50 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels from weekly data act as major support/resistance. 
Breakouts above H3 or below L3, aligned with weekly EMA50 trend and confirmed by volume spikes,
capture strong momentum moves in both bull and bear markets. 12h timeframe targets 12-37 trades/year,
minimizing fee drag while allowing proper trend alignment. Works in bull markets via breakouts 
and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from weekly OHLC
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    weekly_range = df_1w['high'] - df_1w['low']
    camarilla_h3 = df_1w['close'] + 1.1 * weekly_range / 4
    camarilla_l3 = df_1w['close'] - 1.1 * weekly_range / 4
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot levels)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3.values)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Weekly trend filter: price above/below EMA50
        uptrend = ema_50_aligned[i] is not None and curr_close > ema_50_aligned[i]
        downtrend = ema_50_aligned[i] is not None and curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND uptrend AND volume spike
            long_entry = (curr_high > h3_aligned[i]) and uptrend and vol_spike
            # Short: price breaks below L3 AND downtrend AND volume spike
            short_entry = (curr_low < l3_aligned[i]) and downtrend and vol_spike
            
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
            # Exit: price falls below L3 OR loss of trend (price < EMA50)
            if (curr_low < l3_aligned[i]) or (curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 OR loss of trend (price > EMA50)
            if (curr_high > h3_aligned[i]) or (curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0