#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_Volume_V1
Hypothesis: Combines daily Camarilla pivot levels (R1/S1) with 4h trend filter and volume spike.
The strategy targets institutional levels with trend alignment and volume confirmation to capture
strong directional moves while minimizing false breakouts. Uses 1h for entry timing only.
Trades limited to 08:00-20:00 UTC to avoid low-liquidity periods.
Designed for low trade frequency (15-35/year) to reduce fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels R1 and S1 (closer to pivot, more sensitive)
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)  # R1 = C + (H-L)*1.1/12
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)  # S1 = C - (H-L)*1.1/12
    
    # 4h trend filter: EMA20
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to 1h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 20  # EMA20 needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema20 = ema20_4h_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA20 (4h)
            uptrend = close_val > ema20
            downtrend = close_val < ema20
            
            if uptrend and vol_conf:
                # Long: break above R1 with volume
                if close_val > r1:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below S1 with volume
                if close_val < s1:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below R1 or trend reversal
            if close_val < r1:  # Re-enter below R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above S1 or trend reversal
            if close_val > s1:  # Re-enter above S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume_V1"
timeframe = "1h"
leverage = 1.0