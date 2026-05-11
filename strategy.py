#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume
Hypothesis: Uses Camarilla pivot levels (R3/S3) from daily data with EMA34 trend filter and volume confirmation.
Works in bull markets via breakouts and bear markets via mean reversion at pivot levels during low volatility.
Target: 20-40 trades/year to minimize fee drag while capturing strong moves.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (R3, S3) ---
    # Calculate pivot point and ranges
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = close_1d + (range_1d * 1.1 / 2)
    s3_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # --- 1d EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (4h) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume spike threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market trend based on EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Breakout signals at Camarilla levels
        long_breakout = (high[i] > r3_1d_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < s3_1d_aligned[i]) and vol_spike[i]
        
        # Mean reversion signals at Camarilla levels (counter-trend)
        long_reversion = (low[i] <= s3_1d_aligned[i]) and (not price_above_ema)  # Buy at S3 in downtrend
        short_reversion = (high[i] >= r3_1d_aligned[i]) and (not price_below_ema)  # Sell at R3 in uptrend
        
        if position == 0:
            # Look for entry signals
            if long_breakout or long_reversion:
                signals[i] = 0.25
                position = 1
            elif short_breakout or short_reversion:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price reaches S3 (mean reversion target) or breaks below EMA
                exit_signal = (low[i] <= s3_1d_aligned[i]) or (close[i] < ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches R3 (mean reversion target) or breaks above EMA
                exit_signal = (high[i] >= r3_1d_aligned[i]) or (close[i] > ema_34_1d_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals