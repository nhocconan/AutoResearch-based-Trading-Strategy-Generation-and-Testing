#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3
Hypothesis: On 4h timeframe, Donchian channel (20) breakouts capture medium-term momentum.
Break above upper band with volume spike and 1d uptrend (close > EMA34) signals long;
break below lower band with volume spike and 1d downtrend (close < EMA34) signals short.
Uses ATR-based trailing stoploss (3*ATR) and discrete position sizing (0.25) to limit trades 
(~20-50/year) and minimize fee drag. Designed for BTC/ETH to work in both bull and bear 
markets by trading breakouts with trend and volume confirmation, avoiding overtrading 
through tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need Donchian (20) + volume MA (20) + ATR (14) + aligned HTF arrays
    start_idx = max(20, 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume spike and 1d uptrend
            long_breakout = (curr_close > highest_high[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below Donchian lower band with volume spike and 1d downtrend
            short_breakout = (curr_close < lowest_low[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR-based trailing stop: exit if price drops 3*ATR from highest since entry
            if curr_close < highest_since_entry - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: price breaks below Donchian lower band OR trend turns down
            elif (curr_close < lowest_low[i]) or (curr_close < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR-based trailing stop: exit if price rises 3*ATR from lowest since entry
            if curr_close > lowest_since_entry + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Alternative exit: price breaks above Donchian upper band OR trend turns up
            elif (curr_close > highest_high[i]) or (curr_close > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3"
timeframe = "4h"
leverage = 1.0