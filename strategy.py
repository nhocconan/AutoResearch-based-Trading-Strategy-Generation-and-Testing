#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6d_weekly_volatility_breakout_1d_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for volatility and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Weekly volatility breakout: Donchian(4) on weekly high/low
    # Lookback period of 4 weeks (~1 month)
    highest_high_4w = pd.Series(high_1w).rolling(window=4, min_periods=4).max().values
    lowest_low_4w = pd.Series(low_1w).rolling(window=4, min_periods=4).min().values
    
    # Align weekly levels to 6h timeframe
    highest_high_4w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_4w)
    lowest_low_4w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_4w)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly ATR for volatility filter
    tr1_w = pd.Series(high_1w).subtract(pd.Series(low_1w)).abs()
    tr2_w = pd.Series(high_1w).subtract(pd.Series(close_1w).shift(1)).abs()
    tr3_w = pd.Series(low_1w).subtract(pd.Series(close_1w).shift(1)).abs()
    tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
    atr_w = tr_w.rolling(window=4, min_periods=4).mean().values
    atr_ma_w = pd.Series(atr_w).rolling(window=8, min_periods=8).mean().values
    vol_filter_w = atr_w > atr_ma_w  # Only trade when volatility is elevated
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high_4w_aligned[i]) or 
            np.isnan(lowest_low_4w_aligned[i]) or
            np.isnan(vol_filter_w[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly low or trend reverses
            if close[i] < lowest_low_4w_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly high or trend reverses
            if close[i] > highest_high_4w_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs daily EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price breaks above weekly high + uptrend + volatility filter
            if (close[i] > highest_high_4w_aligned[i] and 
                uptrend and 
                vol_filter_w[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly low + downtrend + volatility filter
            elif (close[i] < lowest_low_4w_aligned[i] and 
                  downtrend and 
                  vol_filter_w[i]):
                position = -1
                signals[i] = -0.25
    
    return signals