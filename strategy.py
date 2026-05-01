#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50), volume confirmation (>1.5x 20-bar median volume), and ATR(14) stoploss.
# Long when price breaks above Donchian upper channel AND 1d EMA50 uptrend AND volume spike.
# Short when price breaks below Donchian lower channel AND 1d EMA50 downtrend AND volume spike.
# Exit on opposite Donchian break or ATR stoploss (2*ATR).
# Donchian channels provide clear structure, 1d EMA50 filters for higher-timeframe trend, volume confirms conviction.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_Breakout_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_high).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    highest_since_entry = 0.0  # for long ATR trailing stop
    lowest_since_entry = 0.0   # for short ATR trailing stop
    
    # Start after warmup for Donchian and ATR
    start_idx = max(donchian_window, 14, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian upper AND uptrend AND volume spike
            if curr_close > donchian_high[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price < Donchian lower AND downtrend AND volume spike
            elif curr_close < donchian_low[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, curr_close)
            
            # Exit conditions:
            # 1. Price < Donchian lower (opposite breakout)
            # 2. ATR stoploss: price < highest_since_entry - 2*ATR
            if curr_close < donchian_low[i] or curr_close < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, curr_close)
            
            # Exit conditions:
            # 1. Price > Donchian upper (opposite breakout)
            # 2. ATR stoploss: price > lowest_since_entry + 2*ATR
            if curr_close > donchian_high[i] or curr_close > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals