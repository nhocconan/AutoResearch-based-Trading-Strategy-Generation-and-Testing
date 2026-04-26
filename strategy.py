#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmation
Hypothesis: On 1h timeframe, Camarilla pivot R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation (>1.8x 20-bar avg) capture institutional breakouts. Uses 4h for signal direction (trend + pivot levels) and 1h only for entry timing precision. Session filter (08-20 UTC) reduces noise. Targets 15-30 trades/year to stay within fee drag limits while maintaining edge in both bull and bear markets via trend filter and volatility expansion confirmation.
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
    
    # Get 4h data for HTF trend and Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA50 on 4h for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots on 4h using typical price
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    # Camarilla R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_4h = close_4h + 1.1 * range_4h / 12.0
    camarilla_s1_4h = close_4h - 1.1 * range_4h / 12.0
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours from index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(60, 50, 20)  # 4h lookback, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = vol_val > 1.8 * vol_ma_val
        
        if position == 0 and in_session:
            # Long: price breaks above R1 with uptrend (close > EMA50) and volume confirmation
            long_signal = (high_val > r1_val) and (close_val > ema_50_val) and volume_confirmed
            # Short: price breaks below S1 with downtrend (close < EMA50) and volume confirmation
            short_signal = (low_val < s1_val) and (close_val < ema_50_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 0 and not in_session:
            # Outside session: stay flat
            signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. Price breaks below S1 (opposite Camarilla level)
            if low_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. Price breaks above R1 (opposite Camarilla level)
            if high_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmation"
timeframe = "1h"
leverage = 1.0