#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (2.0x 20-period average).
- Long: price breaks above 12h Camarilla R3 + price > 1d EMA34 + volume > 2.0x 20-period avg volume
- Short: price breaks below 12h Camarilla S3 + price < 1d EMA34 + volume > 2.0x 20-period avg volume
- Exit: trailing stop (2.5x ATR from extreme) OR Camarilla breakout in opposite direction
- Uses 1d EMA34 as trend filter to avoid counter-trend trades and adapt to regime
- Volume confirmation reduces false breakouts (tighter than typical 1.8x)
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: trend filter adapts to regime
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag on 12h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 2.0x 20-period average (tighter spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE before loop for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels (R3, S3) on 12h data
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    hl_range = df_12h['high'] - df_12h['low']
    camarilla_r3 = df_12h['close'] + 1.1 * hl_range * 1.1 / 4
    camarilla_s3 = df_12h['close'] - 1.1 * hl_range * 1.1 / 4
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to LTF (12h -> 12h alignment is identity but keeps pattern consistent)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for volume MA, 14 for ATR, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Break above Camarilla R3
        breakout_down = close[i] < camarilla_s3_aligned[i]  # Break below Camarilla S3
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + price > 1d EMA34 + volume spike
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Camarilla breakout down + price < 1d EMA34 + volume spike
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike:
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
            breakout_down_exit = close[i] < camarilla_s3_aligned[i]
            
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
            breakout_up_exit = close[i] > camarilla_r3_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0