#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend + volume spike + ATR stoploss
    # Long when: price breaks above Donchian(20) upper band AND price > 1d EMA50 AND volume > 1.8x 20-bar avg volume
    # Short when: price breaks below Donchian(20) lower band AND price < 1d EMA50 AND volume > 1.8x 20-bar avg volume
    # Exit when: price touches Donchian(20) midpoint OR adverse 1d EMA50 crossover OR ATR-based stoploss
    # Uses discrete sizing (0.25) targeting 75-200 total trades over 4 years.
    # Donchian provides structure, 1d EMA50 filters counter-trend moves, volume confirms validity, ATR stop manages risk.
    # Works in bull (trend continuation) and bear (mean-reversion at midpoint with tight stops).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint = (highest_high + lowest_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.8x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * avg_volume)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(avg_volume[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar)
        breakout_up = close[i] > highest_high[i]  # break above upper band
        breakout_down = close[i] < lowest_low[i]  # break below lower band
        
        # 1d EMA50 trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (
            close[i] < midpoint[i] or  # price touches midpoint
            not uptrend or             # trend filter fails
            close[i] < (signals[i-1] * ema_50_1d_aligned[i] if i>0 and signals[i-1]!=0 else 0) - 2.5 * atr[i]  # ATR stoploss
        ))
        exit_short = (position == -1 and (
            close[i] > midpoint[i] or  # price touches midpoint
            not downtrend or           # trend filter fails
            close[i] > (signals[i-1] * ema_50_1d_aligned[i] if i>0 and signals[i-1]!=0 else 0) + 2.5 * atr[i]  # ATR stoploss
        ))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_donchian_breakout_ema_volume_atr_v1"
timeframe = "4h"
leverage = 1.0