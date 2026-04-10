#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and 1w EMA trend filter
# - Long when price breaks above Donchian(20) high + ATR(14) > 1.5x ATR(50) 1d + price > EMA(50) 1w
# - Short when price breaks below Donchian(20) low + ATR(14) > 1.5x ATR(50) 1d + price < EMA(50) 1w
# - Exit: price crosses Donchian(10) midpoint (mean reversion)
# - Position sizing: 0.25 discrete level
# - Donchian captures breakouts, ATR filter ensures high conviction moves, EMA filter avoids counter-trend trades
# - Works in bull/bear: breakouts in both directions, EMA filter adapts to trend

name = "4h_1d_1w_donchian_atr_ema_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (highest_high_10 + lowest_low_10) / 2.0
    
    # Calculate 1d ATR(14) and ATR(50) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    atr_ratio = atr_14_1d_aligned / atr_50_1d_aligned
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > 1.5x ATR(50) indicates high conviction moves
        vol_filter = atr_ratio[i] > 1.5
        
        # Trend filter: price relative to 1w EMA(50)
        trend_filter_long = close[i] > ema_50_1w_aligned[i]
        trend_filter_short = close[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout entry conditions
        # Long: price breaks above 20-period high + vol filter + trend filter long
        # Short: price breaks below 20-period low + vol filter + trend filter short
        long_entry = (close[i] > highest_high_20[i] and 
                     vol_filter and 
                     trend_filter_long)
        short_entry = (close[i] < lowest_low_20[i] and 
                      vol_filter and 
                      trend_filter_short)
        
        # Exit conditions: price crosses 10-period Donchian midpoint
        exit_long = close[i] < donchian_mid_10[i]
        exit_short = close[i] > donchian_mid_10[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals