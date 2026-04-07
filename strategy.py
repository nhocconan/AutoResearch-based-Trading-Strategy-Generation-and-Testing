#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour weekly pivot breakout with daily EMA(50) trend filter and volume confirmation
# Uses weekly pivot levels as dynamic support/resistance. Long when price breaks above R1 in daily uptrend,
# short when breaks below S1 in daily downtrend. Weekly pivot provides stronger structural levels than daily.
# Designed for low frequency (target: 15-35 trades/year) to minimize fee drift on 6h timeframe.
# Works in both bull and bear markets by aligning with daily trend - avoids counter-trend trades.

name = "6h_weekly_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly pivot levels (calculated from prior week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance and support levels
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe (use prior week's values)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from daily EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions using weekly pivot levels
        breakout_r1 = close[i] > r1_1w_aligned[i-1] if i > 0 else False
        breakout_s1 = close[i] < s1_1w_aligned[i-1] if i > 0 else False
        breakout_r2 = close[i] > r2_1w_aligned[i-1] if i > 0 else False
        breakout_s2 = close[i] < s2_1w_aligned[i-1] if i > 0 else False
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on S1 breakdown or trend reversal
            if breakout_s1 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on R1 breakout or trend reversal
            if breakout_r1 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long on R1 breakout in uptrend with volume
            if breakout_r1 and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short on S1 breakdown in downtrend with volume
            elif breakout_s1 and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals