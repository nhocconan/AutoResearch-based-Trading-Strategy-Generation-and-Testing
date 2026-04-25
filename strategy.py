#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian channel breakouts capture strong momentum. Combined with 1d EMA50 trend filter and volume confirmation, this strategy works in both bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend). ATR-based stoploss manages risk. Targeting 50-150 total trades over 4 years on 12h timeframe to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Calculate ATR(14) for stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50, Donchian, ATR, volume MA
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > upper Donchian band, above 1d EMA50, volume confirmation
            long_entry = (curr_close > upper_band) and (curr_close > ema_50_val) and volume_confirm
            # Short: price < lower Donchian band, below 1d EMA50, volume confirmation
            short_entry = (curr_close < lower_band) and (curr_close < ema_50_val) and volume_confirm
            
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
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit: stoploss hit OR price crosses below 1d EMA50
            if curr_low <= stop_loss or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit: stoploss hit OR price crosses above 1d EMA50
            if curr_high >= stop_loss or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0