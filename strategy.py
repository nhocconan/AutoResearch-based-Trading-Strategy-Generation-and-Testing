#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and chop regime filter
# Donchian(20) breakout captures momentum in both bull and bear markets
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# Choppiness Index (CHOP) regime filter: CHOP < 38.2 = trending (trade breakouts), CHOP > 61.8 = range (avoid)
# Uses 12h timeframe for lower trade frequency (~15-30 trades/year) and discrete sizing (0.25) to minimize fee drag
# Works in both bull and bear markets by combining price structure (Donchian) with volume and regime filters

name = "12h_Donchian_20_Breakout_VolumeConfirm_ChopFilter_v1"
timeframe = "12h"
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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate Choppiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        chop = np.full_like(close, np.nan)
        for i in range(window, len(close)):
            atr_sum = 0.0
            for j in range(i-window+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            max_high = np.max(high[i-window+1:i+1])
            min_low = np.min(low[i-window+1:i+1])
            if max_high != min_low and atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(window)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 34, 20, 14, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_chop = chop[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop OR price < 1d EMA34 OR chop > 61.8 (range)
            if curr_close < stop_price or curr_close < curr_ema_1d or curr_chop > 61.8:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop OR price > 1d EMA34 OR chop > 61.8 (range)
            if curr_close > stop_price or curr_close > curr_ema_1d or curr_chop > 61.8:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price > Donchian high AND volume spike AND chop < 38.2 (trending)
            if (curr_close > curr_donchian_high and 
                vol_spike and 
                curr_chop < 38.2):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price < Donchian low AND volume spike AND chop < 38.2 (trending)
            elif (curr_close < curr_donchian_low and 
                  vol_spike and 
                  curr_chop < 38.2):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals