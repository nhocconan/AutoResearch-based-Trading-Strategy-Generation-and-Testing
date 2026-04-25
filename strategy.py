#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Volume Confirmation and EMA Filter
Hypothesis: Weekly pivot levels act as strong support/resistance in ranging and trending markets.
Price reverses from weekly R2/S2 with volume confirmation and 1d EMA50 trend filter.
Works in bull via buying S2 reversals, bear via selling R2 reversals. Uses discrete position sizing (0.25)
to control drawdown. Target: 12-37 trades/year on 6h.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot points (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot: P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h (no extra delay needed for pivot points)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Pre-compute volume MA(20) for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_1d_aligned[i]
        r2_level = weekly_r2_aligned[i]
        s2_level = weekly_s2_aligned[i]
        atr_val = atr[i]
        vol_ma = volume_ma_20[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: reversal from S2 with volume spike AND above EMA50 (uptrend bias)
            long_condition = curr_low <= s2_level and curr_close > s2_level and volume_spike and curr_close > ema_50
            # Short: reversal from R2 with volume spike AND below EMA50 (downtrend bias)
            short_condition = curr_high >= r2_level and curr_close < r2_level and volume_spike and curr_close < ema_50
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below EMA50
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above EMA50
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_R2S2_Reversal_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0