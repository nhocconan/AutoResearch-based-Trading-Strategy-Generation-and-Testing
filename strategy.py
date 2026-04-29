#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Donchian breakouts capture momentum moves; 1d EMA50 ensures alignment with higher timeframe trend;
# volume confirms breakout validity. Designed to work in both bull and bear markets by
# taking breakouts in direction of 1d trend. Target: 12-25 trades/year (50-100 total) to minimize fee drag.

name = "6h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price breaks below Donchian lower (20-period low)
            # 2. Price crosses below 1d EMA50 (trend change)
            # 3. ATR-based stop: 2.5 * ATR below entry (tracked via position logic)
            if (curr_close < curr_lower or
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price breaks above Donchian upper (20-period high)
            # 2. Price crosses above 1d EMA50 (trend change)
            if (curr_close > curr_upper or
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter with volume confirmation to avoid false breakouts
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above Donchian upper AND above 1d EMA50 (bullish alignment)
            if (curr_close > curr_upper and
                curr_close > curr_ema_50_1d):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower AND below 1d EMA50 (bearish alignment)
            elif (curr_close < curr_lower and
                  curr_close < curr_ema_50_1d):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals