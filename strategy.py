#!/usr/bin/env python3
"""
4h Camarilla Pivot R3/S3 Breakout with Volume Spike and 1D ADX Filter
Long: Price breaks above R3 + volume > 2x 4h volume MA + ADX > 25
Short: Price breaks below S3 + volume > 2x 4h volume MA + ADX > 25
Exit: Opposite break of S3/R3 respectively
Uses Camarilla levels from daily pivot for institutional reference points
Target: 20-30 trades/year per symbol
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
    
    # Get 1D data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate previous day's Camarilla levels
    # Using prior day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_prev = prev_high - prev_low
    R3 = prev_close + (range_prev * 1.1 / 2)
    S3 = prev_close - (range_prev * 1.1 / 2)
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1D ADX for trend strength filter (avoid choppy markets)
    # Calculate ADX using daily data
    plus_dm = np.where((prev_high[1:] - prev_high[:-1]) > (prev_low[:-1] - prev_low[1:]), 
                       np.maximum(prev_high[1:] - prev_high[:-1], 0), 0)
    minus_dm = np.where((prev_low[:-1] - prev_low[1:]) > (prev_high[1:] - prev_high[:-1]), 
                        np.maximum(prev_low[:-1] - prev_low[1:], 0), 0)
    
    # True Range
    tr1 = prev_high[1:] - prev_low[1:]
    tr2 = np.abs(prev_high[1:] - prev_close[:-1])
    tr3 = np.abs(prev_low[1:] - prev_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
        # Prepend zeros for alignment
        adx_full = np.zeros(len(prev_close))
        adx_full[period+period-1:] = adx[period-1:]
    else:
        adx_full = np.zeros(len(prev_close))
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_full)
    
    # 4h volume moving average for confirmation
    df_4h = get_htf_data(prices, '4h')
    volume_ma_20 = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_aligned[i]
        adx = adx_aligned[i]
        
        if position == 0:
            # Long: break above R3 + volume spike + ADX > 25
            if price > R3_aligned[i] and vol > 2.0 * vol_ma and adx > 25:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S3 + volume spike + ADX > 25
            elif price < S3_aligned[i] and vol > 2.0 * vol_ma and adx > 25:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below S3
            if price < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R3
            if price > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0