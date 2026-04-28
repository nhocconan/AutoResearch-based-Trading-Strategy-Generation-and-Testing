#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter, volume confirmation, and ATR trailing stop
# Long when price breaks above Donchian(20) high + price above 1d EMA(50) + volume > 1.8x 20-bar average
# Short when price breaks below Donchian(20) low + price below 1d EMA(50) + volume > 1.8x 20-bar average
# Exit on ATR(14) trailing stop (3.0 * ATR) or opposite Donchian break
# Target: 20-50 trades/year on 4h to minimize fee drag while capturing strong trends

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume (tight filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    start_idx = max(lookback, 50, 20)  # Donchian(20), 1d EMA(50), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_high = highest_high[i]
        curr_low = lowest_low[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + above 1d EMA50 + volume spike
            if price > curr_high and price > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short entry: price breaks below Donchian low + below 1d EMA50 + volume spike
            elif price < curr_low and price < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - trail stoploss and check for exit
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            
            # Calculate ATR(14) for trailing stop
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = highest_since_entry - 3.0 * atr_val
            
            # Exit conditions: stoploss hit OR price breaks below Donchian low
            if price < stop_loss or price < curr_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - trail stoploss and check for exit
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            
            # Calculate ATR(14) for trailing stop
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = lowest_since_entry + 3.0 * atr_val
            
            # Exit conditions: stoploss hit OR price breaks above Donchian high
            if price > stop_loss or price > curr_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals