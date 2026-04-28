#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend and levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 1d timeframe (same as prices for 1d timeframe)
    ema34_aligned = ema34_1d  # Already aligned since timeframe is 1d
    atr14_aligned = atr14_1d
    
    # Calculate 1-day Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (ATR < 50% of 50-day average)
        atr_ma_50 = pd.Series(atr14_aligned).rolling(window=50, min_periods=50).mean()
        atr_ma_50_values = atr_ma_50.values
        low_vol_filter = not np.isnan(atr_ma_50_values[i]) and atr14_aligned[i] > (0.5 * atr_ma_50_values[i])
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Entry conditions: Donchian breakout with trend alignment and volatility filter
        long_entry = long_breakout and uptrend and low_vol_filter
        short_entry = short_breakout and downtrend and low_vol_filter
        
        # Exit conditions: opposite Donchian breakout or trend reversal
        long_exit = close[i] < lowest_low[i] or not uptrend
        short_exit = close[i] > highest_high[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_Trend_Donchian20_VolFilter"
timeframe = "1d"
leverage = 1.0