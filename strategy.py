#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d ADX14 for trend strength and 1d EMA200 for trend direction,
# 12h Donchian10 breakout for momentum, and volume confirmation. Enters only during 08-20 UTC session.
# Targets 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Uses ADX to filter for trending markets and EMA200 to determine direction.
# Works in bull/bear by following higher timeframe trends and avoiding range-bound markets.
name = "12h_1d_ADX14_EMA200_Donchian10_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX14 and EMA200 (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX14
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d[::-1])[::-1]  # negative of diff(low)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.subtract(high_1d, np.roll(low_1d, 1)))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 12h timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 12h data for Donchian10 breakout (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    # Donchian channels: 10-period high/low
    high_10_12h = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    low_10_12h = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    high_10_12h_aligned = align_htf_to_ltf(prices, df_12h, high_10_12h)
    low_10_12h_aligned = align_htf_to_ltf(prices, df_12h, low_10_12h)
    
    # Volume filter: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(high_10_12h_aligned[i]) or np.isnan(low_10_12h_aligned[i]) or 
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 20 (trending), price above EMA200, breaks 12h Donchian high with volume
            if (adx_14_1d_aligned[i] > 20 and 
                close[i] > ema_200_1d_aligned[i] and 
                close[i] > high_10_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 20 (trending), price below EMA200, breaks 12h Donchian low with volume
            elif (adx_14_1d_aligned[i] > 20 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  close[i] < low_10_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if ADX drops below 20 (ranging) or price breaks below EMA200 or 12h Donchian low
            if (adx_14_1d_aligned[i] < 20 or 
                close[i] < ema_200_1d_aligned[i] or 
                close[i] < low_10_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if ADX drops below 20 (ranging) or price breaks above EMA200 or 12h Donchian high
            if (adx_14_1d_aligned[i] < 20 or 
                close[i] > ema_200_1d_aligned[i] or 
                close[i] > high_10_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals