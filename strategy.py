#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R1S1_Breakout_VolumeATR_Filter_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's data (5 trading days ago)
    # Use rolling window to get prior week's OHLC
    high_5d_ago = np.roll(high_1d, 5)
    low_5d_ago = np.roll(low_1d, 5)
    close_5d_ago = np.roll(close_1d, 5)
    
    # Set first 5 values to NaN (no prior week data)
    high_5d_ago[:5] = np.nan
    low_5d_ago[:5] = np.nan
    close_5d_ago[:5] = np.nan
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    weekly_pivot = (high_5d_ago + low_5d_ago + close_5d_ago) / 3.0
    
    # Calculate R1 and S1: R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - low_5d_ago
    weekly_s1 = 2 * weekly_pivot - high_5d_ago
    
    # Align weekly pivot levels to 1d timeframe (no shift needed as already aligned)
    weekly_pivot_aligned = weekly_pivot
    weekly_r1_aligned = weekly_r1
    weekly_s1_aligned = weekly_s1
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 14-period ATR for volatility (standard for daily)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or \
           np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        ema_trend = ema20_1w_aligned[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above weekly R1 + uptrend + volume
            if price > r1 and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + downtrend + volume
            elif price < s1 and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below weekly pivot or ATR trailing stop
            if price < pivot or price < (high[i] - 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above weekly pivot or ATR trailing stop
            if price > pivot or price > (low[i] + 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals