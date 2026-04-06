#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h Volume + ADX Filter
Hypothesis: Combines Donchian channel breakouts with volume confirmation and ADX trend filter.
In bull markets: long when price breaks above upper band with volume and strong ADX.
In bear markets: short when price breaks below lower band with volume and strong ADX.
Uses 12h timeframe for volume and ADX to reduce noise. Target: 100-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_vol_adx_v1"
timeframe = "4h"
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
    
    # Donchian channel (20-period)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
    
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
    
    # Get 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h ADX calculation (14-period)
    adx_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 14:
        # Calculate +DM and -DM
        plus_dm = np.zeros(len(close_12h))
        minus_dm = np.zeros(len(close_12h))
        tr_12h = np.zeros(len(close_12h))
        
        for i in range(1, len(close_12h)):
            high_diff = high_12h[i] - high_12h[i-1]
            low_diff = low_12h[i-1] - low_12h[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr_12h[i] = max(
                high_12h[i] - low_12h[i],
                np.abs(high_12h[i] - close_12h[i-1]),
                np.abs(low_12h[i] - close_12h[i-1])
            )
        
        # Smooth TR, +DM, -DM (Wilder's smoothing)
        atr_12h = np.zeros(len(close_12h))
        plus_di_12h = np.zeros(len(close_12h))
        minus_di_12h = np.zeros(len(close_12h))
        
        if len(tr_12h) >= 14:
            atr_12h[13] = np.sum(tr_12h[1:14])
            plus_dm_sum = np.sum(plus_dm[1:14])
            minus_dm_sum = np.sum(minus_dm[1:14])
            
            for i in range(14, len(close_12h)):
                atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
                plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
                minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
                
                if atr_12h[i] > 0:
                    plus_di_12h[i] = 100 * plus_dm_sum / atr_12h[i]
                    minus_di_12h[i] = 100 * minus_dm_sum / atr_12h[i]
            
            # Calculate DX and ADX
            dx_12h = np.zeros(len(close_12h))
            for i in range(14, len(close_12h)):
                if plus_di_12h[i] + minus_di_12h[i] > 0:
                    dx_12h[i] = 100 * np.abs(plus_di_12h[i] - minus_di_12h[i]) / (plus_di_12h[i] + minus_di_12h[i])
            
            # Smooth DX to get ADX
            if len(dx_12h) >= 27:  # Need 14 + 13 for ADX
                adx_12h[26] = np.sum(dx_12h[14:27]) / 13
                for i in range(27, len(close_12h)):
                    adx_12h[i] = (adx_12h[i-1] * 13 + dx_12h[i]) / 14
    
    # 12h Volume moving average (20-period)
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        for i in range(19, len(volume_12h)):
            vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align 12h indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for Donchian and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current volume > 1.5x 12h average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # ADX filter: strong trend (ADX > 25)
        adx_filter = adx_aligned[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below lower band OR against ADX trend
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] < lower_band[i] or
                adx_aligned[i] < 20 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above upper band OR against ADX trend
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] > upper_band[i] or
                adx_aligned[i] < 20 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Breakout entries with volume and ADX confirmation
                bull_breakout = close[i] > upper_band[i]
                bear_breakout = close[i] < lower_band[i]
                
                # Long: bullish breakout with volume and strong ADX
                if bull_breakout and volume_filter and adx_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with volume and strong ADX
                elif bear_breakout and volume_filter and adx_filter:
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