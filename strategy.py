#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily Donchian channels (20-period) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Daily volume confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_spike = volume > (vol_ma_aligned * 1.5)
    
    # Daily ATR for volatility filter
    tr1 = pd.Series(df_1d['high'] - df_1d['low'])
    tr2 = pd.Series(np.abs(df_1d['high'] - df_1d['close'].shift(1)))
    tr3 = pd.Series(np.abs(df_1d['low'] - df_1d['close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = 0
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA, daily Donchian, volume, ATR
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema20_1w_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike_val = vol_spike[i]
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high + volume spike + weekly uptrend
            if close[i] > upper_band and vol_spike_val and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: break below Donchian low + volume spike + weekly downtrend
            elif close[i] < lower_band and vol_spike_val and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below Donchian low or trend turns down
            if close[i] < lower_band or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above Donchian high or trend turns up
            if close[i] > upper_band or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume_v2"
timeframe = "1d"
leverage = 1.0