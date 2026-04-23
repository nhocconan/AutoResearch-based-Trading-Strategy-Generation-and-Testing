#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
- Camarilla pivot levels calculated from previous 4h bar: R1 = PP + (H-L)*1.1/12, S1 = PP - (H-L)*1.1/12
- Long: price breaks above R1 + volume > 1.3x 20-period avg + price > 4h EMA50
- Short: price breaks below S1 + volume > 1.3x 20-period avg + price < 4h EMA50
- Exit: price retouches pivot point (PP) or 4h EMA50 trend flip
- Uses Camarilla structure for precise entries, volume for conviction, 4h EMA50 for HTF trend
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.3x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Need at least one 4h bar to calculate pivots
    start_idx = 4  # 4h bar index (each 4h bar = 4 1h bars)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Get previous completed 4h bar for Camarilla calculation
        # Each 4h bar = 4 1h bars, so previous 4h bar ends at index ((i // 4) * 4) - 1
        htf_idx = (i // 4) * 4  # start of current 4h bar
        if htf_idx < 4:  # need at least one completed 4h bar
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        prev_4h_start = htf_idx - 4  # start of previous 4h bar
        prev_4h_end = htf_idx  # end of previous 4h bar (exclusive)
        
        if prev_4h_end > len(high) or prev_4h_start < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate Camarilla levels from previous 4h bar
        phigh = high[prev_4h_start:prev_4h_end].max()
        plow = low[prev_4h_start:prev_4h_end].min()
        pclose = close[prev_4h_end-1]  # close of previous 4h bar
        
        pp = (phigh + plow + pclose) / 3
        range_ = phigh - plow
        r1 = pp + (range_ * 1.1 / 12)
        s1 = pp - (range_ * 1.1 / 12)
        
        # Breakout conditions
        long_breakout = close[i] > r1 and close[i-1] <= r1
        short_breakout = close[i] < s1 and close[i-1] >= s1
        
        # Retouch conditions (exit when price returns to pivot)
        long_exit = close[i] < pp and position == 1
        short_exit = close[i] > pp and position == -1
        
        # EMA trend filter
        long_trend = close[i] > ema_50_4h_aligned[i]
        short_trend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume + trend
            if long_breakout and volume_confirm and long_trend:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S1 + volume + trend
            elif short_breakout and volume_confirm and short_trend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: retouch PP or trend flip
            if long_exit or not long_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: retouch PP or trend flip
            if short_exit or not short_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0