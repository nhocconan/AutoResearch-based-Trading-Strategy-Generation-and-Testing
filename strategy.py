#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels from 1d to identify breakout zones (R3/S3 for reversal, R4/S4 for continuation).
Trades only in direction of 1d EMA34 trend with volume spike confirmation. Designed to capture mean reversion at extreme
levels and breakout momentum in both bull and bear markets by adapting to price action relative to daily pivots.
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = H - L
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    R4 = typical_price + (range_1d * 1.1 / 2)
    R3 = typical_price + (range_1d * 1.1 / 4)
    S3 = typical_price - (range_1d * 1.1 / 4)
    S4 = typical_price - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4.values)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # Calculate volume spike (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Price relative to Camarilla levels
        price_above_R3 = close[i] > R3_aligned[i]
        price_below_S3 = close[i] < S3_aligned[i]
        price_above_R4 = close[i] > R4_aligned[i]
        price_below_S4 = close[i] < S4_aligned[i]
        
        # Entry logic:
        # Long: Price rejects S3/S4 (mean reversion) OR breaks R4 (continuation) in uptrend
        # Short: Price rejects R3/R4 (mean reversion) OR breaks S4 (continuation) in downtrend
        long_entry = vol_confirm and trend_up and (
            (price_below_S3 and close[i] > S3_aligned[i-1]) or  # Rejection of S3
            (price_below_S4 and close[i] > S4_aligned[i-1]) or  # Rejection of S4
            (price_above_R4 and close[i-1] <= R4_aligned[i-1])   # Breakout above R4
        )
        
        short_entry = vol_confirm and trend_down and (
            (price_above_R3 and close[i] < R3_aligned[i-1]) or  # Rejection of R3
            (price_above_R4 and close[i] < R4_aligned[i-1]) or  # Rejection of R4
            (price_below_S4 and close[i-1] >= S4_aligned[i-1])   # Breakout below S4
        )
        
        # Exit logic: Opposite Camarilla level rejection or trend reversal
        long_exit = (price_above_R3 and close[i] < R3_aligned[i-1]) or not trend_up
        short_exit = (price_below_S3 and close[i] > S3_aligned[i-1]) or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0