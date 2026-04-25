#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian breakouts capture strong momentum. 1d EMA34 filters for higher-timeframe trend alignment. 
Volume spike confirms institutional participation. ATR-based stoploss manages risk. 
Designed to work in both bull (breakouts continuation) and bear (failed breakouts reverse) markets.
Targets 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25.
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
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Donchian channels (20-period)
    dc_high = np.full(n, np.nan)
    dc_low = np.full(n, np.nan)
    for i in range(20, n):
        dc_high[i] = np.max(high[i-19:i+1])
        dc_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20)  # 34 for 1d EMA, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_dc = dc_high[i]
        lower_dc = dc_low[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout entries
            # Long breakout: price > upper Donchian + above 1d EMA + volume confirmation
            long_breakout = (curr_close > upper_dc and 
                           curr_close > ema_trend and 
                           volume_confirm)
            # Short breakout: price < lower Donchian + below 1d EMA + volume confirmation
            short_breakout = (curr_close < lower_dc and 
                            curr_close < ema_trend and 
                            volume_confirm)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = atr_val
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # ATR stoploss: exit if price drops below entry - 2.5 * ATR
            stop_price = entry_price - 2.5 * atr_at_entry
            # Exit conditions: stoploss hit OR price retracement to middle of Donchian
            middle_dc = (upper_dc + lower_dc) / 2
            if curr_close <= stop_price or curr_close < middle_dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # ATR stoploss: exit if price rises above entry + 2.5 * ATR
            stop_price = entry_price + 2.5 * atr_at_entry
            # Exit conditions: stoploss hit OR price retracement to middle of Donchian
            middle_dc = (upper_dc + lower_dc) / 2
            if curr_close >= stop_price or curr_close > middle_dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0