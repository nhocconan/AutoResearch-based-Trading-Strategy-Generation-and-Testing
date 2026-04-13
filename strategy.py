#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period) for volatility filter and stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume SMA (20-period) for volume confirmation
    vol_sma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA (50-period) for HTF trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    vol_sma_aligned = align_htf_to_ltf(prices, df_1d, vol_sma)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_sma_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_aligned[i-1]  # break above upper channel
        breakout_down = close[i] < lowest_low_aligned[i-1]  # break below lower channel
        
        # Volume confirmation (current volume > 1.5 * average volume)
        volume_confirm = volume[i] > 1.5 * vol_sma_aligned[i]
        
        # HTF trend filter (price > 1w EMA50 for long, price < 1w EMA50 for short)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_confirm and uptrend and position != 1
        short_entry = breakout_down and volume_confirm and downtrend and position != -1
        
        # Exit conditions: ATR-based stoploss
        exit_long = position == 1 and close[i] <= (signals[i-1] * position_size > 0 and 
                                                (entry_price := highest_high_aligned[i-1]) and 
                                                close[i] <= entry_price - 2.5 * atr_aligned[i])
        exit_short = position == -1 and close[i] >= (signals[i-1] * position_size < 0 and 
                                                   (entry_price := lowest_low_aligned[i-1]) and 
                                                   close[i] >= entry_price + 2.5 * atr_aligned[i])
        
        # Track entry price for stoploss calculation
        if long_entry:
            entry_price = highest_high_aligned[i-1]
        elif short_entry:
            entry_price = lowest_low_aligned[i-1]
        else:
            entry_price = getattr(locals(), 'entry_price', close[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and close[i] <= entry_price - 2.5 * atr_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= entry_price + 2.5 * atr_aligned[i]:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0