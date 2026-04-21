#!/usr/bin/env python3
"""
4h_1d_Keltner_Breakout_Volume_ADX
Hypothesis: Use daily Keltner Channel (ATR-based) breakouts on 4h with volume confirmation and ADX trend filter.
Long when price breaks above upper Keltner band (EMA20 + 2*ATR) with volume > 1.5x average and ADX > 25.
Short when price breaks below lower Keltner band (EMA20 - 2*ATR) with volume > 1.5x average and ADX > 25.
Exit when price crosses back through the 20-period EMA.
Designed for 4h to limit trade frequency (target: 20-50/year) and reduce fee drift.
Keltner channels adapt to volatility, providing robust breakout levels in both trending and volatile markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's data for Keltner Channel (non-lookahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # True Range for ATR calculation
    tr1 = np.abs(prev_high - prev_low)
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) using Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 10)
    
    # EMA(20) of close
    close_series = pd.Series(prev_close)
    ema = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands: EMA20 ± 2*ATR(10)
    upper_keltner = ema + 2 * atr
    lower_keltner = ema - 2 * atr
    
    # Align to 4h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema)
    
    # ADX for regime filter (trending vs ranging) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    atr_4h = wilder_smooth(tr_4h, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_4h != 0, 100 * dm_plus_smooth / atr_4h, 0)
    di_minus = np.where(atr_4h != 0, 100 * dm_minus_smooth / atr_4h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        if position == 0:
            # Long conditions: break above upper Keltner + volume + trending
            if price > upper_keltner_aligned[i] and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower Keltner + volume + trending
            elif price < lower_keltner_aligned[i] and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below EMA20
            if price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above EMA20
            if price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Keltner_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0