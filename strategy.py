#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla R1 (4h) + price > 4h EMA50 + volume > 1.3x 20-period avg
- Short when price breaks below Camarilla S1 (4h) + price < 4h EMA50 + volume > 1.3x 20-period avg
- Exit: ATR trailing stop (2.0x ATR from extreme) OR price reverts to Camarilla pivot point (PP)
- Uses 4h EMA50 as trend filter to align with higher timeframe momentum
- Volume confirmation reduces false signals in ranging markets
- ATR trailing stop manages risk during strong trends
- Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag on 1h timeframe
- Session filter: 08-20 UTC to avoid low-volume Asian session noise
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
    open_time = prices['open_time'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.3x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 4h data ONCE before loop for Camarilla levels and EMA50
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels for 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla pivot point (PP) = (H + L + C) / 3
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla R1 = C + ((H-L) * 1.1/12)
    r1_4h = close_4h + ((high_4h - low_4h) * 1.1 / 12.0)
    # Camarilla S1 = C - ((H-L) * 1.1/12)
    s1_4h = close_4h - ((high_4h - low_4h) * 1.1 / 12.0)
    
    # Align Camarilla levels to 1h timeframe
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for ATR, 50 for EMA50
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(pp_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Volume spike confirmation (> 1.3x average)
        volume_spike = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + price > 4h EMA50 + volume spike
            if close[i] > r1_4h_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
                long_extreme = high[i]
            # Short: price breaks below S1 + price < 4h EMA50 + volume spike
            elif close[i] < s1_4h_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.0x ATR from long extreme (trailing stop)
            # 2. Price reverts to or below Camarilla PP (mean reversion)
            trailing_stop_long = close[i] < long_extreme - 2.0 * atr[i]
            mean_reversion_exit = close[i] <= pp_4h_aligned[i]
            
            if trailing_stop_long or mean_reversion_exit:
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
            # 2. Price reverts to or above Camarilla PP (mean reversion)
            trailing_stop_short = close[i] > short_extreme + 2.0 * atr[i]
            mean_reversion_exit = close[i] >= pp_4h_aligned[i]
            
            if trailing_stop_short or mean_reversion_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_ATRStop"
timeframe = "1h"
leverage = 1.0