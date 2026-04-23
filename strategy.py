#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation (1.5x 24-period average).
- Long: price breaks above Camarilla R1 (4h) + price > 4h EMA50 + volume > 1.5x 24-period avg volume
- Short: price breaks below Camarilla S1 (4h) + price < 4h EMA50 + volume > 1.5x 24-period avg volume
- Exit: trailing stop (1.5x ATR from extreme) OR Camarilla breakout in opposite direction
- Uses 4h EMA50 as trend filter to avoid counter-trend trades and adapt to regime
- Volume confirmation reduces false breakouts
- Session filter: only trade 08-20 UTC to reduce noise
- Position size: 0.20 (discrete to minimize fee churn)
- Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 24-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 4h data ONCE before loop for Camarilla and EMA
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 4h bar
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_r1 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low']) / 12
    camarilla_s1 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low']) / 12
    
    # Calculate EMA50 on 4h close
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14, 50)  # Need 24 for volume MA, 14 for ATR, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous 4h levels)
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above Camarilla R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below Camarilla S1
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla R1 breakout up + price > 4h EMA50 + volume spike
            if breakout_up and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
                long_extreme = high[i]
            # Short: Camarilla S1 breakout down + price < 4h EMA50 + volume spike
            elif breakout_down and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 1.5x ATR from long extreme (trailing stop)
            # 2. Camarilla breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 1.5 * atr[i]
            breakout_down_exit = close[i] < camarilla_s1_aligned[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 1.5x ATR from short extreme (trailing stop)
            # 2. Camarilla breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 1.5 * atr[i]
            breakout_up_exit = close[i] > camarilla_r1_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0