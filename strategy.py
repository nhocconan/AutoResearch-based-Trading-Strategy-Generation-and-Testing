#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use weekly Camarilla pivot levels to identify key support/resistance. Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend following). Filter by weekly EMA trend and volume confirmation to avoid false signals. Works in both bull and bear markets by adapting to market structure - mean reversion in ranging markets, trend following in strong trends. Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Calculate EMA on weekly timeframe for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for pivot calculation
    pp_1w = np.zeros_like(high_1w)
    r1_1w = np.zeros_like(high_1w)
    s1_1w = np.zeros_like(high_1w)
    r2_1w = np.zeros_like(high_1w)
    s2_1w = np.zeros_like(high_1w)
    r3_1w = np.zeros_like(high_1w)
    s3_1w = np.zeros_like(high_1w)
    r4_1w = np.zeros_like(high_1w)
    s4_1w = np.zeros_like(high_1w)
    
    for i in range(1, len(df_1w)):
        # Previous week's values
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        # Pivot point
        pp = (ph + pl + pc) / 3
        pp_1w[i] = pp
        
        # Camarilla levels
        range_ = ph - pl
        r1_1w[i] = pc + (range_ * 1.1 / 12)
        s1_1w[i] = pc - (range_ * 1.1 / 12)
        r2_1w[i] = pc + (range_ * 1.1 / 6)
        s2_1w[i] = pc - (range_ * 1.1 / 6)
        r3_1w[i] = pc + (range_ * 1.1 / 4)
        s3_1w[i] = pc - (range_ * 1.1 / 4)
        r4_1w[i] = pc + (range_ * 1.1 / 2)
        s4_1w[i] = pc - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isclose(pp_aligned[i], 0) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
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