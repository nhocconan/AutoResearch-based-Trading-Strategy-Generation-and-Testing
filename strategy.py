#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel (20-period high/low) from 1d OHLC for institutional breakout zones
# 1w EMA50 ensures alignment with long-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# Works in bull markets (breakout continuation) and bear markets (mean reversion at extremes)
# Focus on BTC and ETH as primary targets

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 2.0x 20-period average (20*1d = 20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 1d Donchian levels (20-period)
    high_1d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_1d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (no shift needed as rolling uses past data)
    # But we need to ensure we don't use current bar's high/low for breakout
    # So we shift the Donchian levels by 1 to avoid look-ahead
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    high_1d_shifted[0] = np.nan  # First value has no previous bar
    low_1d_shifted[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and Donchian)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_1d_shifted[i]) or np.isnan(low_1d_shifted[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above 20-day high AND price > 1w EMA50 (bullish trend) AND volume spike
            if (close[i] > high_1d_shifted[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-day low AND price < 1w EMA50 (bearish trend) AND volume spike
            elif (close[i] < low_1d_shifted[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below 20-day low (mean reversion) OR price below 1w EMA50 (trend change)
            if close[i] < low_1d_shifted[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above 20-day high (mean reversion) OR price above 1w EMA50 (trend change)
            if close[i] > high_1d_shifted[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals