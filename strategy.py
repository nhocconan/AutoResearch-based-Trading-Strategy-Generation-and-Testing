#!/usr/bin/env python3
"""
6h ADX + SuperTrend Confluence with Volume Confirmation
Hypothesis: Combining ADX trend strength (>25) with SuperTrend direction captures strong momentum moves.
Volume confirmation (>1.5x 20-period average) filters weak breakouts. Works in bull (long when ADX>25, SuperTrend up, volume spike) 
and bear (short when ADX>25, SuperTrend down, volume spike). Designed for 6h timeframe with tight entry conditions
to achieve 12-37 trades/year. Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades.
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
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for higher timeframe trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ADX (14-period) on 6h data
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate SuperTrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    atr_st = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr_st
    basic_lb = (high + low) / 2 - multiplier * atr_st
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, n):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # SuperTrend direction
    supertrend = np.zeros(n)
    supertrend[0] = final_ub[0]
    for i in range(1, n):
        if supertrend[i-1] == final_ub[i-1]:
            if close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    supertrend_direction = np.where(close > supertrend, 1, -1)  # 1: uptrend, -1: downtrend
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ADX and SuperTrend
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(supertrend_direction[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx[i]
        st_direction = supertrend_direction[i]
        htf_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: ADX>25 (strong trend), SuperTrend up, price above HTF EMA, volume spike
            long_entry = (adx_val > 25) and (st_direction == 1) and (curr_close > htf_trend) and vol_spike
            # Short: ADX>25 (strong trend), SuperTrend down, price below HTF EMA, volume spike
            short_entry = (adx_val > 25) and (st_direction == -1) and (curr_close < htf_trend) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: ADX<20 (weak trend) OR SuperTrend flips down OR price crosses below HTF EMA
            if (adx_val < 20) or (st_direction == -1) or (curr_close < htf_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: ADX<20 (weak trend) OR SuperTrend flips up OR price crosses above HTF EMA
            if (adx_val < 20) or (st_direction == 1) or (curr_close > htf_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_SuperTrend_Confluence_VolumeSpike"
timeframe = "6h"
leverage = 1.0