#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d ADX(14) Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian 20-period breakouts on 12h chart capture significant price moves.
Trend filtered by 1d ADX > 25 ensures we only trade in strong trending markets (works in both bull and bear).
Volume spike confirms breakout authenticity. ATR-based stoploss manages risk.
Designed for 12h timeframe targeting 12-35 trades/year. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 1d data for ADX trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM and -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # Positive when low decreases
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate True Range
    tr1 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing periods (Wilder's smoothing)
    period = 14
    def wilders_smoothing(values, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current_value) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Smooth TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, (plus_dm_smooth / atr_1d) * 100, 0)
    minus_di_1d = np.where(atr_1d != 0, (minus_dm_smooth / atr_1d) * 100, 0)
    
    # Calculate DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     np.abs((plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)) * 100, 0)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian(20) channels on 12h data
    if len(close) >= 20:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        for i in range(n):
            start_idx = max(0, i - 19)
            donchian_high[i] = np.max(high[start_idx:i+1])
            donchian_low[i] = np.min(low[start_idx:i+1])
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Calculate ATR(14) for stoploss on 12h data
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ADX, Donchian, ATR, and volume MA to propagate
    start_idx = max(34, 20, 14, 20)  # ADX needs ~34 bars (14+20 for smoothing)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian channel AND strong trend AND volume spike
            long_condition = (curr_close > upper_channel) and strong_trend and volume_spike
            # Short: price breaks below lower Donchian channel AND strong trend AND volume spike
            short_condition = (curr_close < lower_channel) and strong_trend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below lower channel (reversal)
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above upper channel (reversal)
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dADX14_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0