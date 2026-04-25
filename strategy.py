#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and aligned with 1d EMA34 trend 
capture momentum moves. Works in bull markets (long H3 breaks) and bear markets (short L3 breaks) 
by requiring 1d EMA34 alignment. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. 
Target: 20-40 trades/year on 4h.
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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Range = high - low
    rang = high - low
    
    # Camarilla levels (based on previous day's typical price and range)
    # H3 = typical_price + (rang * 1.1 / 4)
    # L3 = typical_price - (rang * 1.1 / 4)
    # We need previous day's values, so shift by 1
    prev_typical = pd.Series(typical_price).shift(1).values
    prev_rang = pd.Series(rang).shift(1).values
    
    H3 = prev_typical + (prev_rang * 1.1 / 4)
    L3 = prev_typical - (prev_rang * 1.1 / 4)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = 20  # for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume_spike = volume_spike[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > H3[i]) and curr_volume_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < L3[i]) and curr_volume_spike and (curr_close < ema_trend)
            
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
            # Exit: price falls back below H3 (failed breakout) OR price < 1d EMA34 (trend change)
            if (curr_close < H3[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises back above L3 (failed breakdown) OR price > 1d EMA34 (trend change)
            if (curr_close > L3[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0