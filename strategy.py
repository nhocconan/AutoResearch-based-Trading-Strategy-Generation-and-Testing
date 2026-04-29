#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1w EMA50 trend filter and volume confirmation
# Long: Close > Donchian(20) high AND price > 1w EMA50 AND volume > 1.8x 20-bar avg
# Short: Close < Donchian(20) low AND price < 1w EMA50 AND volume > 1.8x 20-bar avg
# Exit: Close crosses Donchian(10) midpoint OR price crosses 1w EMA50 OR ATR stoploss
# ATR stoploss: 2.5 * ATR(15) from entry price
# Donchian channels provide clear trend structure that works in both bull and bear markets
# 1w EMA50 filter ensures we trade with the primary trend, reducing whipsaws
# Volume confirmation filters out low-participation breakouts
# Target: 100-180 total trades over 4 years (25-45/year) on 4h timeframe
# Discrete position sizing: 0.30 for long/short, 0.0 for flat to balance return and drawdown

name = "4h_Donchian_Breakout_1wEMA50_VolumeSpike_ATRStop_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 15-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 15)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Calculate Donchian channels (20-period for entry, 10-period for exit)
        if i >= 20:
            donch_high_20 = np.max(high[i-20:i])
            donch_low_20 = np.min(low[i-20:i])
        else:
            signals[i] = 0.0
            continue
            
        if i >= 10:
            donch_high_10 = np.max(high[i-10:i])
            donch_low_10 = np.min(low[i-10:i])
            donch_mid = (donch_high_10 + donch_low_10) / 2
        else:
            donch_mid = close[i]
        
        curr_close = close[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * curr_atr
            # Exit conditions: Close below Donchian(10) midpoint OR price below 1w EMA50 OR stoploss hit
            if curr_close < donch_mid or curr_close < curr_ema_1w or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * curr_atr
            # Exit conditions: Close above Donchian(10) midpoint OR price above 1w EMA50 OR stoploss hit
            if curr_close > donch_mid or curr_close > curr_ema_1w or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Close > Donchian(20) high AND price > 1w EMA50 AND volume spike
            if (curr_close > donch_high_20 and 
                curr_close > curr_ema_1w and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Close < Donchian(20) low AND price < 1w EMA50 AND volume spike
            elif (curr_close < donch_low_20 and 
                  curr_close < curr_ema_1w and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals