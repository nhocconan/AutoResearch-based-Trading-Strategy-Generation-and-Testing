#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Long: price breaks above Camarilla R3 (1h) + price > 4h EMA34 + volume > 1.5x 20-period avg volume
- Short: price breaks below Camarilla S3 (1h) + price < 4h EMA34 + volume > 1.5x 20-period avg volume
- Exit: trailing stop (2.0x ATR from extreme) OR Camarilla breakout in opposite direction
- Uses Camarilla pivots for precise intraday levels, 4h EMA34 for trend filter to avoid counter-trend trades
- Volume confirmation reduces false breakouts
- Target: 15-37 trades/year (60-150 total over 4 years) on 1h timeframe to minimize fee drag
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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivots on 1h data (primary timeframe)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # Using previous bar's high/low/close for current bar's levels
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * rng
    camarilla_s3 = prev_close - 1.1 * rng
    
    # Load 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for Camarilla, 14 for ATR, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > camarilla_r3[i]  # Break above Camarilla R3
        breakout_down = close[i] < camarilla_s3[i]  # Break below Camarilla S3
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + price > 4h EMA34 + volume spike
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
                long_extreme = high[i]
            # Short: Camarilla breakout down + price < 4h EMA34 + volume spike
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Camarilla breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            breakout_down_exit = close[i] < camarilla_s3[i]
            
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
            # 1. Price reverses 2.0x ATR from short extreme (trailing stop)
            # 2. Camarilla breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            breakout_up_exit = close[i] > camarilla_r3[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike_ATRStop"
timeframe = "1h"
leverage = 1.0