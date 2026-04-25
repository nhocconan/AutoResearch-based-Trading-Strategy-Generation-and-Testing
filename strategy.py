#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) and volume confirmation (>1.5x 20-bar avg) capture institutional moves with favorable risk-reward. Uses ATR-based stop (2.0) and discrete sizing (0.25). Weekly pivot acts as regime filter: long bias above pivot, short bias below. Targets 12-25 trades/year by requiring confluence of structure, regime, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    # Use previous completed weekly bar to avoid look-ahead
    prev_weekly_high = np.concatenate([[np.nan], high_1w[:-1]])
    prev_weekly_low = np.concatenate([[np.nan], low_1w[:-1]])
    prev_weekly_close = np.concatenate([[np.nan], close_1w[:-1]])
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get daily data for additional trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 6h
    # Use rolling window on actual 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 6h for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_6h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 50, 14, 20)  # Donchian, EMA50, ATR, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_6h[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        dh = donchian_high[i]
        dl = donchian_low[i]
        wp = weekly_pivot_aligned[i]
        ema1d = ema_50_1d_aligned[i]
        atr_val = atr_6h[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Regime filters:
        # 1. Weekly pivot direction: long bias above weekly pivot, short bias below
        long_bias = close_val > wp
        short_bias = close_val < wp
        
        # 2. Daily EMA50 trend filter: align with higher timeframe trend
        uptrend = close_val > ema1d
        downtrend = close_val < ema1d
        
        # Volume confirmation: moderate spike to avoid noise
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with regime and volume
            # Long: price breaks above Donchian HIGH with long bias and uptrend
            long_signal = (high_val > dh) and long_bias and uptrend and volume_confirm
            # Short: price breaks below Donchian LOW with short bias and downtrend
            short_signal = (low_val < dl) and short_bias and downtrend and volume_confirm
            
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
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Breakdown below Donchian LOW (failed breakout)
            elif close_val < dl:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Loss of weekly pivot bias (regime change)
            elif close_val < wp:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Breakout above Donchian HIGH (failed breakdown)
            elif close_val > dh:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Loss of weekly pivot bias (regime change)
            elif close_val > wp:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0