#!/usr/bin/env python3
"""
6h Elder Ray + ADX Regime + Volume Spike
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength. 
ADX > 25 confirms trending market. Volume spike validates institutional participation. 
Long when Bull Power > 0 and rising, ADX > 25, volume spike. 
Short when Bear Power > 0 and rising, ADX > 25, volume spike. 
Works in bull (buy strength) and bear (sell weakness) via symmetric logic. 
Target 15-25 trades/year on 6h to avoid fee drag.
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
    
    # Get 1d data for EMA13 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = pd.Series(df_1d['close'])
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Bull Power and Bear Power from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate ADX(14) from 1d data for trend strength
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        plus_dm = np.zeros(len(high_arr))
        minus_dm = np.zeros(len(high_arr))
        tr = np.zeros(len(high_arr))
        
        for i in range(1, len(high_arr)):
            plus_dm[i] = max(0, high_arr[i] - high_arr[i-1])
            minus_dm[i] = max(0, low_arr[i-1] - low_arr[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros(len(high_arr))
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high_arr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high_arr))
        minus_di = np.zeros(len(high_arr))
        dx = np.zeros(len(high_arr))
        
        for i in range(period, len(high_arr)):
            plus_di[i] = 100 * (plus_dm[i] / atr[i]) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm[i] / atr[i]) if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros(len(high_arr))
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high_arr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA13, ADX, ATR, volume MA
    start_idx = max(13, 2*14, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
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
        ema_13_val = ema_13_1d_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Elder Ray conditions
        bull_power_rising = bull_power_val > 0 and (i == start_idx or bull_power_val > bull_power_aligned[i-1])
        bear_power_rising = bear_power_val > 0 and (i == start_idx or bear_power_val > bear_power_aligned[i-1])
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: Bull Power > 0 and rising, strong trend, volume spike
            long_entry = bull_power_rising and strong_trend and volume_confirm
            # Short: Bear Power > 0 and rising, strong trend, volume spike
            short_entry = bear_power_rising and strong_trend and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: Bull Power turns negative OR 2.5*ATR trailing stop OR ADX weakens
            if bull_power_val <= 0 or curr_close < (highest_since_entry - 2.5 * atr_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: Bear Power turns negative OR 2.5*ATR trailing stop OR ADX weakens
            if bear_power_val <= 0 or curr_close > (lowest_since_entry + 2.5 * atr_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0