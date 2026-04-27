#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Uses tighter entry (R1/S1) to reduce trade frequency vs R3/S3 variants. 12h EMA50 ensures alignment with higher timeframe trend.
Volume spike confirms breakout authenticity. Designed for 20-40 trades/year on 4h to minimize fee drag.
Works in both bull and bear markets by following 12h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Camarilla levels (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr0 = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need enough for ATR, EMA50, volume average
    start_idx = max(100, 50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 12h trend with volume spike
            # Long: price breaks above Camarilla R1 AND 12h trend is up (close > EMA50) AND volume spike
            # Short: price breaks below Camarilla S1 AND 12h trend is down (close < EMA50) AND volume spike
            long_breakout = close_val > camarilla_r1[i]
            short_breakout = close_val < camarilla_s1[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
            if long_breakout and trend_up and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_breakout and trend_down and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Long - update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit when: price breaks below Camarilla S1 (failed breakout) OR ATR trailing stop hit
            if close_val < camarilla_s1[i] or close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit when: price breaks above Camarilla R1 (failed breakout) OR ATR trailing stop hit
            if close_val > camarilla_r1[i] or close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0