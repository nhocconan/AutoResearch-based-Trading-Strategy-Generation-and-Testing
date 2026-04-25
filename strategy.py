#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 + EMA34 uptrend + volume > 1.5x average volume.
Short when price breaks below S1 + EMA34 downtrend + volume > 1.5x average volume.
Exit on opposite level touch or trend reversal.
Position size: 0.25 to limit drawdown. Target: 20-40 trades/year on 4h.
Uses proven Camarilla structure with volume and trend filters to reduce false breakouts.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume (20-period) for volume filter
    vol_ma = np.zeros(n)
    vol_ma[:] = np.nan
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels from previous day's OHLC
        # Need previous day's data - use 1d data shifted by 1
        if i >= 1440:  # Approximate bars per day for 4h (24*60/4 = 360, but using index)
            # Get previous completed 1d bar
            prev_1d_idx = len(df_1d) - 1 - ((len(prices) - i - 1) // 360)
            if prev_1d_idx >= 1:
                ph = df_1d['high'].iloc[prev_1d_idx]
                pl = df_1d['low'].iloc[prev_1d_idx]
                pc = df_1d['close'].iloc[prev_1d_idx]
                
                # Camarilla levels
                R1 = pc + (ph - pl) * 1.1 / 12
                S1 = pc - (ph - pl) * 1.1 / 12
                
                # Determine 1d HTF trend (bullish = price above EMA34)
                htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
                htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
                
                # Volume confirmation
                volume_confirm = volume[i] > 1.5 * vol_ma[i]
                
                if position == 0:
                    # Long setup: price breaks above R1 + uptrend + volume
                    long_setup = (close[i] > R1) and htf_1d_bullish and volume_confirm
                    
                    # Short setup: price breaks below S1 + downtrend + volume
                    short_setup = (close[i] < S1) and htf_1d_bearish and volume_confirm
                    
                    if long_setup:
                        signals[i] = 0.25
                        position = 1
                    elif short_setup:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                elif position == 1:
                    # Long: hold position
                    signals[i] = 0.25
                    # Exit: price touches S1 (stop) OR 1d trend turns bearish
                    if (close[i] <= S1) or (not htf_1d_bullish):
                        signals[i] = 0.0
                        position = 0
                elif position == -1:
                    # Short: hold position
                    signals[i] = -0.25
                    # Exit: price touches R1 (stop) OR 1d trend turns bullish
                    if (close[i] >= R1) or (htf_1d_bullish):
                        signals[i] = 0.0
                        position = 0
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
        else:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0