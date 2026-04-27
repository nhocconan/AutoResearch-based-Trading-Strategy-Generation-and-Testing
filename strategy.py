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
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily ATR14 for volatility
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr1.iloc[0] = 0
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_d_aligned = align_htf_to_ltf(prices, df_1d, atr14_d)
    
    # Daily volume average
    vol_avg_d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_d)
    
    # 6h ATR for stop calculation
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr6 = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA, daily ATR/volume, 6h ATR
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(atr14_d_aligned[i]) or np.isnan(vol_avg_d_aligned[i]) or np.isnan(atr6[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema20_1w_aligned[i]
        daily_atr = atr14_d_aligned[i]
        daily_vol_avg = vol_avg_d_aligned[i]
        current_vol = volume[i]
        atr6_val = atr6[i]
        
        # Volume spike condition: current volume > 1.5 * daily average
        vol_spike = current_vol > (daily_vol_avg * 1.5)
        
        if position == 0:
            # Long: close above weekly EMA + volatility expansion + volume spike
            if close[i] > weekly_trend and atr6_val > (daily_atr * 1.2) and vol_spike:
                signals[i] = size
                position = 1
            # Short: close below weekly EMA + volatility expansion + volume spike
            elif close[i] < weekly_trend and atr6_val > (daily_atr * 1.2) and vol_spike:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below weekly EMA or volatility contraction
            if close[i] < weekly_trend or atr6_val < (daily_atr * 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above weekly EMA or volatility contraction
            if close[i] > weekly_trend or atr6_val < (daily_atr * 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Volatility_Expansion_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0