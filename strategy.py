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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily ATR (14-period) for position sizing and stoploss
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(np.abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = 0  # First TR is zero due to shift
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily Donchian channels (20-period) for breakout signals
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA50, daily Donchian, volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr_14_aligned[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema50_1w_aligned[i]
        upper_donchian = high_20[i]
        lower_donchian = low_20[i]
        vol_spike_val = vol_spike[i]
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume spike + weekly uptrend
            if close[i] > upper_donchian and vol_spike_val and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + volume spike + weekly downtrend
            elif close[i] < lower_donchian and vol_spike_val and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below lower Donchian or weekly trend turns down
            if close[i] < lower_donchian or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above upper Donchian or weekly trend turns up
            if close[i] > upper_donchian or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0