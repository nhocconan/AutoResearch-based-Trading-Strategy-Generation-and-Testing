#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR trailing stop
# Long: Close > Donchian Upper(20) AND price > 1d EMA50 AND volume > 1.8x 20-bar avg
# Short: Close < Donchian Lower(20) AND price < 1d EMA50 AND volume > 1.8x 20-bar avg
# Exit: ATR trailing stop (3.0 * ATR from extreme) OR Donchian midpoint reversion
# Uses 1d HTF for stable trend filter to avoid whipsaws in choppy markets
# Volume confirmation filters low-participation breakouts
# Discrete position sizing: ±0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRTrail_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for trailing stop (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    start_idx = max(50, 20, 30)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Donchian channels (20-period)
        if i >= 20:
            donch_high = np.max(high[i-20:i])
            donch_low = np.min(low[i-20:i])
            donch_mid = (donch_high + donch_low) / 2.0
        else:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and trailing stop
        if position == 1:  # Long position
            highest_high = max(highest_high, high[i])
            stop_price = highest_high - 3.0 * curr_atr
            # Exit: price below Donchian midpoint OR trailing stop hit
            if curr_close < donch_mid or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            lowest_low = min(lowest_low, low[i])
            stop_price = lowest_low + 3.0 * curr_atr
            # Exit: price above Donchian midpoint OR trailing stop hit
            if curr_close > donch_mid or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > Donchian Upper AND price > 1d EMA50 AND volume spike
            if (curr_close > donch_high and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high = high[i]
            # Short entry: Close < Donchian Lower AND price < 1d EMA50 AND volume spike
            elif (curr_close < donch_low and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low = low[i]
            else:
                signals[i] = 0.0
    
    return signals