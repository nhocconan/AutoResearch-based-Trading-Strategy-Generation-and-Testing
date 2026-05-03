#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when: price breaks above 4h Donchian upper channel AND close > 1d EMA50 AND volume > 1.5x 24-bar average
# Short when: price breaks below 4h Donchian lower channel AND close < 1d EMA50 AND volume > 1.5x 24-bar average
# Exit via ATR(20) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses 4h Donchian for structure (proven edge from top performers), 1d EMA50 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (based on previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower (20-period) from completed 4h bars
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (no additional delay needed for Donchian)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Calculate 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 1d timeframe (no additional delay needed for EMA)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate volume average (24-bar)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = 0
    lowest_low = 0
    
    for i in range(20, n):
        # Skip if HTF data not yet available
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(ema_50_aligned[i]):
            continue
            
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # Entry conditions
        long_entry = (close[i] > donchian_upper_aligned[i] and 
                     close[i] > ema_50_aligned[i] and 
                     vol_confirm)
        short_entry = (close[i] < donchian_lower_aligned[i] and 
                      close[i] < ema_50_aligned[i] and 
                      vol_confirm)
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:
            highest_high = max(highest_high, high[i])
            long_exit = close[i] < highest_high - 2.0 * atr[i]
        elif position == -1:
            lowest_low = min(lowest_low, low[i])
            short_exit = close[i] > lowest_low + 2.0 * atr[i]
        
        # Generate signal
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals