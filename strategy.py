#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout with 4h EMA34 Trend Filter and 1d Volume Spike
Hypothesis: Camarilla R1/S1 breakouts on 1h with daily volume spike and 4h EMA34 trend alignment capture swing moves in both bull/bear markets. Uses discrete position sizing (0.20) and session filter (08-20 UTC) to target 60-150 total trades over 4 years, minimizing fee drag while maintaining edge across regimes.
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
    
    # ATR for trailing stop
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average (daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # 4h EMA34 trend filter (MTF) - loaded ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1h Camarilla pivot levels (R1/S1 for breakout)
    camarilla_r1 = close + (high - low) * 1.1 / 12
    camarilla_s1 = close - (high - low) * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Start index: need enough for all indicators
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions: price breaks Camarilla R1/S1 levels
        breakout_long = curr_close > camarilla_r1[i]
        breakout_short = curr_close < camarilla_s1[i]
        
        if position == 0:
            # Look for entry signals - require: R1/S1 breakout + volume spike + 4h EMA34 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_34_4h_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_34_4h_aligned[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_high
            elif short_entry:
                signals[i] = -0.20
                position = -1
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management: ATR trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            exit_level = highest_since_entry - (2.5 * atr_14[i])
            
            if curr_close < exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management: ATR trailing stop
            lowest_since_entry = min(lowest_since_entry, curr_low)
            exit_level = lowest_since_entry + (2.5 * atr_14[i])
            
            if curr_close > exit_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0