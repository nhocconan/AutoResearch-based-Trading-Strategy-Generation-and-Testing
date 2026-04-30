#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike + ATR stoploss.
# Long when price breaks above Donchian upper (20-bar high) and price > 1d EMA34 and volume > 2.0x 20-bar avg.
# Short when price breaks below Donchian lower (20-bar low) and price < 1d EMA34 and volume > 2.0x 20-bar avg.
# Exit via ATR-based trailing stop: long exit when price < highest high since entry - 2.5*ATR(20),
# short exit when price > lowest low since entry + 2.5*ATR(20).
# Donchian channels provide clear structure, 1d EMA34 filters for higher-timeframe trend,
# volume spike confirms breakout strength, ATR stop manages risk in both bull and bear markets.
# Timeframe: 4h as per experiment guidelines.

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for Donchian, EMA34, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_20[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donch_high = donch_high[i]
        curr_donch_low = donch_low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, price > 1d EMA34, volume spike
            if (curr_close > curr_donch_high and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_bar = i
                highest_since_entry = curr_close
            # Short: price breaks below Donchian lower, price < 1d EMA34, volume spike
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_bar = i
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # ATR trailing stop: exit when price < highest_since_entry - 2.5*ATR
            if curr_close < (highest_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # ATR trailing stop: exit when price > lowest_since_entry + 2.5*ATR
            if curr_close > (lowest_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals