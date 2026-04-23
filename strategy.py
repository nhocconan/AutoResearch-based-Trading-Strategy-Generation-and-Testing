#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation (2.0x 20-period average).
- Long: price breaks above 1d Camarilla R1 + price > 1w EMA50 + volume > 2.0x 20-period avg volume
- Short: price breaks below 1d Camarilla S1 + price < 1w EMA50 + volume > 2.0x 20-period avg volume
- Exit: ATR trailing stop (2.5x ATR from extreme) OR Camarilla breakout in opposite direction
- Uses 1w EMA50 as trend filter to avoid counter-trend trades and adapt to regime
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: trend filter adapts to regime
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag on 1d timeframe
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
    
    # Volume confirmation: > 2.0x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels on 1d data
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    camarilla_s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for ATR, 50 for EMA
    
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
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above Camarilla R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below Camarilla S1
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + price > 1w EMA50 + volume spike
            if breakout_up and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Camarilla breakout down + price < 1w EMA50 + volume spike
            elif breakout_down and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Camarilla breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
            breakout_down_exit = close[i] < camarilla_s1_aligned[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from short extreme (trailing stop)
            # 2. Camarilla breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            breakout_up_exit = close[i] > camarilla_r1_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_1wEMA50_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0