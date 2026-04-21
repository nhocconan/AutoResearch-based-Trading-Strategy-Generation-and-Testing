#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align weekly trend to daily
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR(14) for volatility and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema10_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema10_weekly = ema10_1w_aligned[i]
        atr_val = atr[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_ma_val = vol_ma[i]
        vol = volume[i]
        price = close[i]
        
        # Trend filter: weekly EMA10 slope
        if i >= 51:
            ema10_prev = ema10_1w_aligned[i-1]
            weekly_uptrend = ema10_weekly > ema10_prev
            weekly_downtrend = ema10_weekly < ema10_prev
        else:
            weekly_uptrend = weekly_downtrend = False
        
        # Volume filter: above average
        vol_filter = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above upper Donchian + weekly uptrend + volume
            if price > upper_channel and weekly_uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + weekly downtrend + volume
            elif price < lower_channel and weekly_downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Donchian channel or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below lower channel
                if price < lower_channel:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above upper channel
                if price > upper_channel:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA10_Trend_Volume"
timeframe = "1d"
leverage = 1.0