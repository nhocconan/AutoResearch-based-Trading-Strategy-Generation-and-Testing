#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout with 1d EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance. A breakout above H3 or below L3 on 12h timeframe,
confirmed by 1d EMA50 trend direction and volume spikes, captures significant momentum moves. Designed for 12h to target
12-37 trades/year (50-150 over 4 years) by requiring confluence of Camarilla breakout, 1d EMA50 trend, and volume confirmation,
reducing overtrading and fee drag while working in both bull and bear regimes.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 12h bar (H4, L4, H3, L3)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    daily_range = high - low
    # Camarilla levels based on previous bar
    H4 = close + (daily_range * 1.1 / 2)
    L4 = close - (daily_range * 1.1 / 2)
    H3 = close + (daily_range * 1.1 / 4)
    L3 = close - (daily_range * 1.1 / 4)
    # Shift by 1 to use previous bar's levels (avoid look-ahead)
    H4 = np.roll(H4, 1)
    L3 = np.roll(L3, 1)
    H3 = np.roll(H3, 1)
    L4 = np.roll(L4, 1)
    H4[0] = np.nan
    L3[0] = np.nan
    H3[0] = np.nan
    L4[0] = np.nan
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and Camarilla calculation
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Camarilla breakout + trend + volume
            # Long: price breaks above H3 AND bullish bias AND volume spike
            long_entry = (curr_high > H3[i]) and bullish_bias and vol_spike
            # Short: price breaks below L3 AND bearish bias AND volume spike
            short_entry = (curr_low < L3[i]) and bearish_bias and vol_spike
            
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
            # Exit: price falls below L3 (mean reversion) OR loss of bullish bias
            if (curr_low < L3[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (mean reversion) OR loss of bearish bias
            if (curr_high > H3[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0