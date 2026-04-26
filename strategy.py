#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with 12h EMA34 trend filter and volume confirmation (>1.8x 20-bar avg) capture institutional breakouts. Uses Camarilla levels from 1d (more reliable than intraday) and requires volume spike to avoid false breakouts. Trend filter ensures we trade with the 12h momentum. Designed for low trade frequency (15-25/year) to minimize fee drag while working in both bull and bear markets via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    r4 = close_1d + 1.5 * camarilla_range
    s4 = close_1d - 1.5 * camarilla_range
    
    # Align Camarilla levels (no extra delay needed as they're based on completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA34 on 12h for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume average (20-period = ~5 days on 6h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(30, 34, 20)  # 1d lookback, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = vol_val > 1.8 * vol_ma_val
        
        if position == 0:
            # Long: break above R3 with uptrend (close > EMA34) and volume confirmation
            # But not a break above R4 (avoid exhaustion gaps)
            long_signal = (high_val > r3_val) and (close_val > ema_34_val) and volume_confirmed and (high_val <= r4_val)
            # Short: break below S3 with downtrend (close < EMA34) and volume confirmation
            # But not a break below S4 (avoid exhaustion gaps)
            short_signal = (low_val < s3_val) and (close_val < ema_34_val) and volume_confirmed and (low_val >= s4_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price breaks below S3 (failed breakout)
            if close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Break above R4 (take profit on strong move)
            elif high_val >= r4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price breaks above R3 (failed breakout)
            if close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Break below S4 (take profit on strong move)
            elif low_val <= s4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0