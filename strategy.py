#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via smoothed median price alignment
# In bull markets: JAW > TEETH > LIPS (uptrend). In bear markets: JAW < TEETH < LIPS (downtrend)
# 1w EMA50 ensures we only trade with the higher timeframe major trend
# Volume confirmation (1.5x 24-period average) filters weak breakouts
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d data (smoother for 12h trading)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Median price = (high + low) / 2
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Jaw (13-period SMMA of median price, shifted 8 bars forward)
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (8-period SMMA of median price, shifted 5 bars forward)
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (5-period SMMA of median price, shifted 3 bars forward)
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50, 24)  # warmup for EMA50, Alligator, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume confirmation
            if curr_volume_confirm:
                # Bullish entry: JAW > TEETH > LIPS (uptrend) AND price above 1w EMA50
                if curr_jaw > curr_teeth and curr_teeth > curr_lips and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: JAW < TEETH < LIPS (downtrend) AND price below 1w EMA50
                elif curr_jaw < curr_teeth and curr_teeth < curr_lips and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Alligator turns bearish (JAW < TEETH < LIPS)
            if curr_jaw < curr_teeth and curr_teeth < curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator turns bullish (JAW > TEETH > LIPS)
            if curr_jaw > curr_teeth and curr_teeth > curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals