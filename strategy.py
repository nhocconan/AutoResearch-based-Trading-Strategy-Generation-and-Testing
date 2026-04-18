#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for price channel and filters
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first period
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = ema_50_aligned[i] > ema_50_aligned[i-1]  # rising EMA
        downtrend = ema_50_aligned[i] < ema_50_aligned[i-1]  # falling EMA
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > 0.5 * np.mean(atr_14_aligned[max(0, i-20):i+1])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > upper_20_aligned[i]
        breakdown_down = close[i] < lower_20_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + volatility + breakout above upper channel
            if uptrend and vol_confirm and vol_filter and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + volatility + breakdown below lower channel
            elif downtrend and vol_confirm and vol_filter and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal, volatility drop, or breakdown
            if not uptrend or not vol_filter or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal, volatility drop, or breakout
            if not downtrend or not vol_filter or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA50_Volume_Volatility"
timeframe = "12h"
leverage = 1.0