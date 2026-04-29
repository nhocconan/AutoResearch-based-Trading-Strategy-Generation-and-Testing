#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA34 trend filter, volume confirmation, and ATR-based stoploss
# Donchian(20) upper/lower breakout captures momentum bursts
# 1d EMA34 filters for higher-timeframe trend alignment
# Volume spike (>2.0x 20-bar average) confirms breakout validity
# ATR stoploss limits downside during choppy/false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_Donchian_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # max(20, 14, 34) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Handle existing positions
        if position == 1:  # Long position
            # Exit conditions: Donchian lower breakout OR ATR stoploss OR trend change
            if (curr_low <= lowest_low_20[i] or  # Donchian lower break (exit long)
                curr_close <= entry_price - 2.0 * atr[i] or  # ATR stoploss
                curr_close < ema_34_1d_aligned[i]):  # Price below 1d EMA34 (trend change)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit conditions: Donchian upper breakout OR ATR stoploss OR trend change
            if (curr_high >= highest_high_20[i] or  # Donchian upper break (exit short)
                curr_close >= entry_price + 2.0 * atr[i] or  # ATR stoploss
                curr_close > ema_34_1d_aligned[i]):  # Price above 1d EMA34 (trend change)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: Donchian upper breakout AND price above 1d EMA34 AND volume spike
            if (curr_high >= highest_high_20[i] and 
                curr_close > ema_34_1d_aligned[i] and
                vol_spike):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: Donchian lower breakout AND price below 1d EMA34 AND volume spike
            elif (curr_low <= lowest_low_20[i] and 
                  curr_close < ema_34_1d_aligned[i] and
                  vol_spike):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals