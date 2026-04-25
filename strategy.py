#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian breakouts capture strong momentum moves. Using 1w EMA34 as trend filter ensures we only trade in the direction of the weekly trend. Volume spike confirms institutional participation. ATR-based stoploss limits downside. Works in bull/bear markets by trend-filtering breakout signals.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for ATR(14) stoploss
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian(20) channels on 12h
    # We need to calculate 12h high/low from the prices DataFrame
    # Since we're on 12h timeframe, we can use rolling window directly
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_multiplier = 2.5
    
    # Start index: need enough for Donchian(20) + EMA34 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            if position != 0:
                # Check stoploss
                if position == 1 and close[i] < entry_price - atr_multiplier * atr_14_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > entry_price + atr_multiplier * atr_14_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_val = atr_14_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above upper Donchian channel AND above 1w EMA34 (uptrend) AND volume spike
            long_condition = (curr_high > upper_channel) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below lower Donchian channel AND below 1w EMA34 (downtrend) AND volume spike
            short_condition = (curr_low < lower_channel) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long: trail stoploss or exit on trend reversal
            # Update stoploss to highest close since entry minus ATR*multiplier
            if curr_close > entry_price:
                entry_price = max(entry_price, curr_close)  # trailing entry for stoploss calculation
            
            if curr_close < entry_price - atr_multiplier * atr_val or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short: trail stoploss or exit on trend reversal
            # Update stoploss to lowest close since entry plus ATR*multiplier
            if curr_close < entry_price:
                entry_price = min(entry_price, curr_close)  # trailing entry for stoploss calculation
            
            if curr_close > entry_price + atr_multiplier * atr_val or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0