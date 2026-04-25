#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian breakouts capture strong momentum moves. 
Filtered by 1d EMA34 trend for direction and volume confirmation to avoid false breakouts.
ATR-based stoploss manages risk. Designed to work in both bull (breakouts continue) 
and bear (breakouts fail quickly via stoploss) markets by limiting losing trades.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period ATR for stoploss (using 12h data)
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First TR
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.mean(tr[i-19:i+1])
    
    # Calculate 20-period Donchian channels (using 12h data)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel AND price > EMA34 (uptrend) AND volume confirmation
            long_entry = (curr_high > upper_channel and 
                         curr_close > ema_trend and volume_confirm)
            # Short: price breaks below lower Donchian channel AND price < EMA34 (downtrend) AND volume confirmation
            short_entry = (curr_low < lower_channel and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below lower Donchian channel OR stoploss hit
            stoploss_level = entry_price - 2.5 * atr_val
            if (curr_close < lower_channel or curr_close < stoploss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above upper Donchian channel OR stoploss hit
            stoploss_level = entry_price + 2.5 * atr_val
            if (curr_close > upper_channel or curr_close > stoploss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0