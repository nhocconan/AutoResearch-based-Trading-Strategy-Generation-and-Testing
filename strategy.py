#!/usr/bin/env python3
"""
1h Camarilla R1S1 Breakout with 4h EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) on 1h act as intraday support/resistance.
Breakout above R1 or below S1 with 4h EMA34 trend alignment and volume confirmation
captures momentum moves. Uses 4h for signal direction (trend filter) and 1h for entry timing.
Session filter (08-20 UTC) reduces noise. Targets 15-37 trades/year to avoid fee drag.
Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakouts below S1 in downtrend).
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
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 4h EMA alignment
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 1h bar using prior bar's OHLC
        if i == 0:
            continue
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        rang = prev_high - prev_low
        
        # Avoid division by zero in case of doji or flat bar
        if rang <= 0:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Camarilla R1 and S1 levels (more sensitive than R3/S3 for 1h)
        r1 = prev_close + rang * 1.1 / 12
        s1 = prev_close - rang * 1.1 / 12
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 4h EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above R1 AND uptrend AND volume spike
            long_entry = (curr_close > r1) and uptrend and vol_spike
            # Short: break below S1 AND downtrend AND volume spike
            short_entry = (curr_close < s1) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below S1 (reversal) OR loss of uptrend
            if (curr_close < s1) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price breaks above R1 (reversal) OR loss of downtrend
            if (curr_close > r1) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSp"
timeframe = "1h"
leverage = 1.0