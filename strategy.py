#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum. Volume confirmation ensures institutional participation.
ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions. Symmetric logic for long/short.
Target 20-30 trades/year on 4h to avoid fee drag. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for ADX trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr_4h = np.zeros(len(df_4h))
    for i in range(1, len(df_4h)):
        tr_4h[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
    
    # Directional Movement
    plus_dm_4h = np.zeros(len(df_4h))
    minus_dm_4h = np.zeros(len(df_4h))
    for i in range(1, len(df_4h)):
        up_move = high_4h[i] - high_4h[i-1]
        down_move = low_4h[i-1] - low_4h[i]
        plus_dm_4h[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_4h[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_period = 14
    atr_4h = WilderSmoothing(tr_4h, tr_period)
    plus_dm_smooth = WilderSmoothing(plus_dm_4h, tr_period)
    minus_dm_smooth = WilderSmoothing(minus_dm_4h, tr_period)
    
    # Directional Indicators
    plus_di_4h = np.full_like(atr_4h, np.nan)
    minus_di_4h = np.full_like(atr_4h, np.nan)
    dx_4h = np.full_like(atr_4h, np.nan)
    for i in range(tr_period, len(atr_4h)):
        if atr_4h[i] != 0:
            plus_di_4h[i] = 100 * (plus_dm_smooth[i] / atr_4h[i])
            minus_di_4h[i] = 100 * (minus_dm_smooth[i] / atr_4h[i])
            di_sum = plus_di_4h[i] + minus_di_4h[i]
            if di_sum != 0:
                dx_4h[i] = 100 * abs(plus_di_4h[i] - minus_di_4h[i]) / di_sum
    
    # ADX: smoothed DX
    adx_4h = WilderSmoothing(dx_4h, tr_period)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate Donchian(20) on primary timeframe (4h)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian, volume MA, ADX
    start_idx = max(20, 20, 28)  # 28 for ADX (14+14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_4h_aligned[i]
        vol_ma = vol_ma_20[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Donchian levels
            # Long: price breaks above upper band with volume confirmation in trending market
            long_breakout = (curr_close > upper_band) and volume_confirm and trending
            # Short: price breaks below lower band with volume confirmation in trending market
            short_breakout = (curr_close < lower_band) and volume_confirm and trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below lower band OR 2.0*ATR trailing stop OR ADX < 20 (trend weak)
            atr_val = np.mean([tr_4h[i]] * 0 + [abs(high[i] - close[i-1]), abs(low[i] - close[i-1])])  # current TR
            atr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if curr_close < lower_band or curr_close < (highest_since_entry - 2.0 * atr_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above upper band OR 2.0*ATR trailing stop OR ADX < 20 (trend weak)
            atr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if curr_close > upper_band or curr_close > (lowest_since_entry + 2.0 * atr_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0