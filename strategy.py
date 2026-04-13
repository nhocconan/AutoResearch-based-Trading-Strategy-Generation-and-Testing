#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr[i] = np.nanmean(tr[i-13:i+1])  # Simple average of last 14 TR
    
    # Calculate 20-period EMA on 1d
    ema = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema[i] = (close_1d[i] * 2 / (20 + 1)) + (ema[i-1] * (20 - 1) / (20 + 1))
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period SMA on 1w
    sma_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        sma_1w[i] = np.mean(close_1w[i-20:i])
    
    # Align indicators to daily timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(ema_aligned[i]) or
            np.isnan(sma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly SMA
        uptrend = close[i] > sma_1w_aligned[i]
        downtrend = close[i] < sma_1w_aligned[i]
        
        # Mean reversion signals: price deviation from daily EMA
        price_deviation = (close[i] - ema_aligned[i]) / atr_aligned[i]
        
        # Entry conditions: mean reversion in trending market
        long_entry = uptrend and price_deviation < -1.0  # Pullback in uptrend
        short_entry = downtrend and price_deviation > 1.0  # Bounce in downtrend
        
        # Exit conditions: price returns to EMA or trend reversal
        exit_long = position == 1 and (close[i] >= ema_aligned[i] or not uptrend)
        exit_short = position == -1 and (close[i] <= ema_aligned[i] or not downtrend)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "1d_mean_reversion_trend_filter"
timeframe = "1d"
leverage = 1.0