#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6h timeframe, use Camarilla pivot levels from daily timeframe to identify key support/resistance. Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend following). Filter by daily EMA trend and volume confirmation to avoid false signals. Works in both bull and bear markets by adapting to market structure - mean reversion in ranging markets, trend following in strong trends. Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA on daily timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for pivot calculation
    pp_1d = np.zeros_like(high_1d)
    r1_1d = np.zeros_like(high_1d)
    s1_1d = np.zeros_like(high_1d)
    r2_1d = np.zeros_like(high_1d)
    s2_1d = np.zeros_like(high_1d)
    r3_1d = np.zeros_like(high_1d)
    s3_1d = np.zeros_like(high_1d)
    r4_1d = np.zeros_like(high_1d)
    s4_1d = np.zeros_like(high_1d)
    
    for i in range(1, len(df_1d)):
        # Previous day's values
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point
        pp = (ph + pl + pc) / 3
        pp_1d[i] = pp
        
        # Camarilla levels
        range_ = ph - pl
        r1_1d[i] = pc + (range_ * 1.1 / 12)
        s1_1d[i] = pc - (range_ * 1.1 / 12)
        r2_1d[i] = pc + (range_ * 1.1 / 6)
        s2_1d[i] = pc - (range_ * 1.1 / 6)
        r3_1d[i] = pc + (range_ * 1.1 / 4)
        s3_1d[i] = pc - (range_ * 1.1 / 4)
        r4_1d[i] = pc + (range_ * 1.1 / 2)
        s4_1d[i] = pc - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isclose(pp_aligned[i], 0) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion failure) or above R4 (take profit)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion failure) or below S4 (take profit)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Mean reversion fade at R3/S3
                if close[i] > r3_aligned[i] and below_ema:
                    # Fade rejection at R3 - go short
                    position = -1
                    signals[i] = -0.25
                elif close[i] < s3_aligned[i] and above_ema:
                    # Fade rejection at S3 - go long
                    position = 1
                    signals[i] = 0.25
                # Breakout continuation at R4/S4
                elif close[i] > r4_aligned[i] and above_ema:
                    # Break above R4 with trend - go long
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i] and below_ema:
                    # Break below S4 with trend - go short
                    position = -1
                    signals[i] = -0.25
    
    return signals