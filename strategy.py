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
    volume = prices['volume'].values
    
    # Get weekly data for weekly Donchian
    df_1w = get_htf_data(prices, '1w')
    # Weekly Donchian channels (20 weeks)
    weekly_high = df_1w['high'].rolling(window=20, min_periods=20).max()
    weekly_low = df_1w['low'].rolling(window=20, min_periods=20).min()
    weekly_high_arr = weekly_high.values
    weekly_low_arr = weekly_low.values
    # Align to daily timeframe with proper delay
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_arr)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_arr)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_threshold[i])):
            continue
            
        # Long conditions:
        # 1. Price breaks above weekly Donchian high
        # 2. Weekly close above weekly EMA50 (uptrend)
        # 3. Volume confirmation
        # 4. ATR > 0 (sufficient volatility)
        if (close[i] > weekly_high_aligned[i] and 
            weekly_ema50_aligned[i] > 0 and  # EMA calculated
            volume[i] > vol_threshold[i] and
            atr[i] > 0):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian low
        # 2. Weekly close below weekly EMA50 (downtrend)
        # 3. Volume confirmation
        # 4. ATR > 0
        elif (close[i] < weekly_low_aligned[i] and 
              weekly_ema50_aligned[i] > 0 and  # EMA calculated
              volume[i] > vol_threshold[i] and
              atr[i] > 0):
            signals[i] = -0.25
            
        # Exit conditions:
        # Exit long when price breaks below weekly EMA50
        elif signals[i-1] == 0.25 and weekly_ema50_aligned[i] > 0 and close[i] < weekly_ema50_aligned[i]:
            signals[i] = 0.0
            
        # Exit short when price breaks above weekly EMA50
        elif signals[i-1] == -0.25 and weekly_ema50_aligned[i] > 0 and close[i] > weekly_ema50_aligned[i]:
            signals[i] = 0.0
            
        # Otherwise hold position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian20_EMA50_Volume"
timeframe = "1d"
leverage = 1.0