#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly volume confirmation and volatility filter
# Uses weekly Donchian channel (20 weeks) as trend filter, daily Donchian (5 days) for entry
# Breakouts traded only when weekly volume > 1.5x average and ATR < 0.03*price (low volatility)
# Designed to work in both bull (breakouts up) and bear (breakouts down) with controlled trade frequency
# Timeframe: 1d, HTF: 1w

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly Donchian channel (20-period) for trend filter
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly average volume for confirmation
    avg_vol_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    
    # Daily Donchian channel (5-period) for entry signals
    high_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(avg_vol_1w_aligned[i])):
            continue
        
        # Calculate current price as percentage of ATR for volatility filter
        atr_percent = atr_1w_aligned[i] / close[i] if close[i] > 0 else 1
        
        # Long entry: price breaks above weekly Donchian high + volume confirmation + low volatility
        if (close[i] > high_20_aligned[i] and
            volume[i] > 1.5 * avg_vol_1w_aligned[i] and
            atr_percent < 0.03 and  # Low volatility filter
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly Donchian low + volume confirmation + low volatility
        elif (close[i] < low_20_aligned[i] and
              volume[i] > 1.5 * avg_vol_1w_aligned[i] and
              atr_percent < 0.03 and  # Low volatility filter
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or volatility spike
        elif position == 1 and (close[i] < low_20_aligned[i] or atr_percent > 0.05):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_20_aligned[i] or atr_percent > 0.05):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Volume_Volatility"
timeframe = "1d"
leverage = 1.0