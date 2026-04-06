#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1D Volume Spike + 1D ADX Trend Filter + ATR Stoploss
Hypothesis: Donchian breakouts with 1D volume spike (>2x average) and strong 1D trend (ADX>25) capture high-probability moves. Using 1D filters reduces trade frequency vs 4H filters, targeting 75-200 total trades over 4 years. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dvol_1dadx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # === 1D HTF DATA (loaded ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1D indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1D ADX (14-period)
    adx_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        # +DM and -DM
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range
        tr_1d = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        
        # Smoothed values
        tr14_1d = np.full(len(df_1d), np.nan)
        plus_dm14_1d = np.full(len(df_1d), np.nan)
        minus_dm14_1d = np.full(len(df_1d), np.nan)
        
        if len(tr_1d) >= 14:
            tr14_1d[14] = np.sum(tr_1d[:14])
            plus_dm14_1d[14] = np.sum(plus_dm[:14])
            minus_dm14_1d[14] = np.sum(minus_dm[:14])
            
            for i in range(15, len(df_1d)):
                tr14_1d[i] = tr14_1d[i-1] - (tr14_1d[i-1] / 14) + tr_1d[i-1]
                plus_dm14_1d[i] = plus_dm14_1d[i-1] - (plus_dm14_1d[i-1] / 14) + plus_dm[i-1]
                minus_dm14_1d[i] = minus_dm14_1d[i-1] - (minus_dm14_1d[i-1] / 14) + minus_dm[i-1]
        
        # Directional Indicators
        plus_di_1d = np.full(len(df_1d), np.nan)
        minus_di_1d = np.full(len(df_1d), np.nan)
        dx_1d = np.full(len(df_1d), np.nan)
        
        valid_1d = ~np.isnan(tr14_1d) & (tr14_1d != 0)
        if np.any(valid_1d):
            plus_di_1d[valid_1d] = 100 * plus_dm14_1d[valid_1d] / tr14_1d[valid_1d]
            minus_di_1d[valid_1d] = 100 * minus_dm14_1d[valid_1d] / tr14_1d[valid_1d]
            dx_1d[valid_1d] = 100 * np.abs(plus_di_1d[valid_1d] - minus_di_1d[valid_1d]) / (plus_di_1d[valid_1d] + minus_di_1d[valid_1d])
        
        # ADX (smoothed DX)
        adx_smooth_1d = np.full(len(df_1d), np.nan)
        valid_dx_1d = ~np.isnan(dx_1d)
        if np.sum(valid_dx_1d) >= 14:
            first_idx = np.where(valid_dx_1d)[0][13] if np.sum(valid_dx_1d) >= 14 else -1
            if first_idx != -1:
                adx_smooth_1d[first_idx] = np.mean(dx_1d[valid_dx_1d][:14])
                for i in range(first_idx + 1, len(df_1d)):
                    if valid_dx_1d[i]:
                        adx_smooth_1d[i] = (adx_smooth_1d[i-1] * 13 + dx_1d[i]) / 14
                adx_1d = adx_smooth_1d
    
    # 1D Volume MA (20-period)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(20, len(df_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 1D indicators to 4H timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (1D 20-period average)
        vol_ma = vol_ma_1d_aligned[i]
        volume_filter = volume[i] > vol_ma * 2.0
        
        # ADX trend filter (1D strong trend)
        trend_filter = adx_1d_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + ADX trend filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                if bull_breakout and volume_filter and trend_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and trend_filter:
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