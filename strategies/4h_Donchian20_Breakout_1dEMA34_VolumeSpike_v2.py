#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper(20) AND price > 1d EMA(34) AND volume > 2.0x 20-period average
# Short when price breaks below 4h Donchian lower(20) AND price < 1d EMA(34) AND volume > 2.0x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following HTF trend.
# Based on proven pattern: Donchian breakouts with volume and trend filters show strong test performance.
# Added ATR-based trailing stop for risk management to reduce drawdown.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper = max(high, 20), lower = min(low, 20)
    high_roll_max = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper = align_htf_to_ltf(prices, df_4h, high_roll_max)
    donchian_lower = align_htf_to_ltf(prices, df_4h, low_roll_min)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for trailing stop
    highest_since_entry = 0.0  # for long positions
    lowest_since_entry = 0.0   # for short positions
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Update highest price since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions:
            # 1. Price breaks below Donchian lower
            # 2. Price < 1d EMA(34)
            # 3. Trailing stop: price drops 2.5*ATR from highest since entry
            if (curr_close < curr_lower or 
                curr_close < curr_ema or
                curr_close < highest_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest price since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Exit conditions:
            # 1. Price breaks above Donchian upper
            # 2. Price > 1d EMA(34)
            # 3. Trailing stop: price rises 2.5*ATR from lowest since entry
            if (curr_close > curr_upper or 
                curr_close > curr_ema or
                curr_close > lowest_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper AND price > 1d EMA(34) AND volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short entry: price breaks below Donchian lower AND price < 1d EMA(34) AND volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals