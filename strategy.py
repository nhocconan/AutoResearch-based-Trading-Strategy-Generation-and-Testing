#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + Daily EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3 (resistance) and L3 (support) levels from daily pivot act as 
intraday magnets. Breakouts beyond these levels with daily EMA34 trend alignment and 
volume spike capture institutional participation. Works in bull markets (trend 
continuation) and bear markets (failed breaks, mean reversion to H3/L3). 12h timeframe 
targets 12-37 trades/year to avoid fee drag.
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
    open_ = prices['open'].values
    
    # Daily data for Camarilla pivot and EMA34 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels: based on previous day's OHLC
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    
    # Align to 12h timeframe (wait for completed daily bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for daily data (pivot + EMA) and volume MA
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > camarilla_h3_aligned[i]
        breakout_short = curr_close < camarilla_l3_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + daily EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on retrace below L3 or trend change
            if curr_close < camarilla_l3_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on retrace above H3 or trend change
            if curr_close > camarilla_h3_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0