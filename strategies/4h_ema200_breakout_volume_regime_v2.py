#!/usr/bin/env python3
# 4h_ema200_breakout_volume_regime_v2
# Hypothesis: Price breaking above/below 200-period EMA on 4h with volume > 2x average and ADX > 25 (trending regime)
# captures strong momentum moves in both bull and bear markets. Uses ADX to avoid chop, volume to confirm strength,
# and 200EMA as dynamic support/resistance. Target: 20-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema200_breakout_volume_regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA200 on 4h
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for trend regime (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smooth(tr, 14)
    plus_dm_smoothed = wilders_smooth(plus_dm, 14)
    minus_dm_smoothed = wilders_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
    minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200  # Need EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema200[i]) or np.isnan(adx[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below EMA200 OR ADX < 20 (trend weakening)
            if close[i] < ema200[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above EMA200 OR ADX < 20 (trend weakening)
            if close[i] > ema200[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 2x average volume
            volume_ok = volume[i] > 2 * avg_volume[i]
            # Trend regime: ADX > 25
            trend_ok = adx[i] > 25
            
            # Long entry: Price above EMA200 with volume and trend
            if close[i] > ema200[i] and volume_ok and trend_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below EMA200 with volume and trend
            elif close[i] < ema200[i] and volume_ok and trend_ok:
                position = -1
                signals[i] = -0.25
    
    return signals