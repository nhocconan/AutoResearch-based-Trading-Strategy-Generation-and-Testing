#!/usr/bin/env python3
"""
1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation + ATR stoploss
Hypothesis: Donchian channel breakouts from weekly EMA50 trend filter with volume confirmation
provides robust edge in both bull and bear markets. 1d timeframe targets 7-25 trades/year.
Works in bull via breakout continuation, bear via mean-reversion from extreme levels when
trend aligns. Uses proven patterns from top performers with tight entry conditions.
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
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    donchian_high = np.full(len(daily_high), np.nan)
    donchian_low = np.full(len(daily_low), np.nan)
    
    for i in range(len(daily_high)):
        if i >= 19:
            start_idx = i - 19
            donchian_high[i] = np.max(daily_high[start_idx:i+1])
            donchian_low[i] = np.min(daily_low[start_idx:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        ema_50_1w = np.full(n, np.nan)
    else:
        close_1w = df_1w['close'].values
        ema_50_1w_vals = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w = align_htf_to_ltf(prices, df_1w, ema_50_1w_vals)
    
    # Calculate ATR(14) for stoploss on 1d data
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
    
    # Start index: need enough for Donchian, EMA50_1w, ATR, and volume MA to propagate
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w[i]) or 
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
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        ema50_1w = ema_50_1w[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper channel AND above weekly EMA50 AND volume spike
            long_condition = (curr_close > upper_channel) and (curr_close > ema50_1w) and volume_spike
            # Short: price breaks below lower channel AND below weekly EMA50 AND volume spike
            short_condition = (curr_close < lower_channel) and (curr_close < ema50_1w) and volume_spike
            
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

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0