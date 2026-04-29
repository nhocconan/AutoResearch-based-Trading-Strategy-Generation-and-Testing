#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long: Close > Donchian Upper(20) AND price > 1d EMA50 AND volume > 1.8x 24-bar avg
# Short: Close < Donchian Lower(20) AND price < 1d EMA50 AND volume > 1.8x 24-bar avg
# Exit: Close crosses Donchian midpoint OR price crosses 1d EMA50 OR ATR stoploss (2.5x ATR)
# Donchian channels provide structural breakouts that work in both bull and bear markets
# EMA50 filter ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters out false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Discrete position sizing: 0.25 for long/short, 0.0 for flat to minimize fee churn

name = "12h_Donchian_Breakout_1dEMA50_VolumeConfirmation_ATRStop_v1"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Donchian channels (20-period)
        if i >= 20:
            # Donchian Upper: highest high of last 20 bars (including current)
            donchian_upper = np.max(high[i-19:i+1])
            # Donchian Lower: lowest low of last 20 bars (including current)
            donchian_lower = np.min(low[i-19:i+1])
            # Donchian Midpoint: average of upper and lower
            donchian_mid = (donchian_upper + donchian_lower) / 2.0
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume confirmation: current volume > 1.8x 24-period average
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-23:i+1])
        else:
            vol_ma_24 = 0.0
        vol_confirm = volume[i] > 1.8 * vol_ma_24 if vol_ma_24 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * curr_atr
            # Exit conditions: Close below Donchian midpoint OR price below 1d EMA50 OR stoploss hit
            if curr_close < donchian_mid or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * curr_atr
            # Exit conditions: Close above Donchian midpoint OR price above 1d EMA50 OR stoploss hit
            if curr_close > donchian_mid or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > Donchian Upper AND price > 1d EMA50 AND volume confirmation
            if (curr_close > donchian_upper and 
                curr_close > curr_ema_1d and
                vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Close < Donchian Lower AND price < 1d EMA50 AND volume confirmation
            elif (curr_close < donchian_lower and 
                  curr_close < curr_ema_1d and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals