#!/usr/bin/env python3
"""
4h Camarilla H3L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels from daily data act as significant intraday support/resistance. 
Breakouts above H3 or below L3, aligned with daily EMA34 trend and confirmed by volume spikes,
capture strong momentum moves in both bull and bear markets. 4h timeframe targets 20-50 trades/year,
minimizing fee drag while allowing for proper trend alignment.
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from daily OHLC
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    daily_range = df_1d['high'] - df_1d['low']
    camarilla_h3 = df_1d['close'] + 1.1 * daily_range / 4
    camarilla_l3 = df_1d['close'] - 1.1 * daily_range / 4
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed for pivot levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Daily trend filter: price above/below EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND uptrend AND volume spike
            long_entry = (curr_high > h3_aligned[i]) and uptrend and vol_spike
            # Short: price breaks below L3 AND downtrend AND volume spike
            short_entry = (curr_low < l3_aligned[i]) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below L3 OR loss of trend (price < EMA34)
            if (curr_low < l3_aligned[i]) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 OR loss of trend (price > EMA34)
            if (curr_high > h3_aligned[i]) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0