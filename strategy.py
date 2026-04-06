#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w volume confirmation and ADX trend filter
Hypothesis: Donchian breakouts on daily timeframe capture institutional momentum, 
filtered by weekly volume confirmation for conviction and ADX for trend strength.
Works in bull (buy breakouts above rising ADX) and bear (sell breakdowns above rising ADX).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_vol_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothing
        atr = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        
        if n >= period + 1:
            # Initial values
            atr[period] = np.mean(tr[:period])
            plus_dm_sum = np.sum(plus_dm[:period])
            minus_dm_sum = np.sum(minus_dm[:period])
            
            plus_di[period] = 100 * plus_dm_sum / atr[period] if atr[period] != 0 else 0
            minus_di[period] = 100 * minus_dm_sum / atr[period] if atr[period] != 0 else 0
            
            # Wilder smoothing
            for i in range(period + 1, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
                plus_di[i] = 100 * (plus_di[i-1] * (period - 1) + plus_dm[i-1]) / (atr[i] * period) if atr[i] != 0 else 0
                minus_di[i] = 100 * (minus_di[i-1] * (period - 1) + minus_dm[i-1]) / (atr[i] * period) if atr[i] != 0 else 0
        
        # DX and ADX
        dx = np.full(n, np.nan)
        adx = np.full(n, np.nan)
        
        for i in range(period, n):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        if n >= 2 * period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # 20-period average volume on 1w
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(20, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    
    # Align volume MA to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Donchian channels (20-period) from 1d data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 1d volume > 1.5x 1w average volume (scaled)
        # Scale 1w volume to 1d: approx 1/5 of 1w volume (since 5x 1d in 1w)
        vol_threshold = vol_ma_1w_aligned[i] / 5.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX weak (< 20)
            # Stoploss: price drops 2*ATR below entry
            atr_val = np.mean(np.maximum(
                high[max(0, i-14):i] - low[max(0, i-14):i],
                np.maximum(
                    np.abs(high[max(0, i-14):i] - close[max(0, i-14)-1:i-1]),
                    np.abs(low[max(0, i-14):i] - close[max(0, i-14)-1:i-1])
                )
            )) if i >= 14 else 0
            
            if (close[i] < lower[i] or
                adx[i] < 20 or
                close[i] < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX weak (< 20)
            # Stoploss: price rises 2*ATR above entry
            atr_val = np.mean(np.maximum(
                high[max(0, i-14):i] - low[max(0, i-14):i],
                np.maximum(
                    np.abs(high[max(0, i-14):i] - close[max(0, i-14)-1:i-1]),
                    np.abs(low[max(0, i-14):i] - close[max(0, i-14)-1:i-1])
                )
            )) if i >= 14 else 0
            
            if (close[i] > upper[i] or
                adx[i] < 20 or
                close[i] > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                # Breakout entries: upper/lower with ADX > 25 (trending market)
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                strong_trend = adx[i] > 25
                
                # Long: breakout above upper with strong trend + volume
                if bull_breakout and strong_trend and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with strong trend + volume
                elif bear_breakout and strong_trend and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals