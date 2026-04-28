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
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d weekly pivot levels (monthly close for weekly pivot)
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().values
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    range_hl = weekly_high - weekly_low
    
    # Weekly resistance/support levels (similar to Camarilla but weekly)
    R1 = pivot + (range_hl * 1.1 / 2)
    S1 = pivot - (range_hl * 1.1 / 2)
    R2 = pivot + (range_hl * 1.1)
    S2 = pivot - (range_hl * 1.1)
    
    # Align weekly levels to daily
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Get weekly data for volume and volatility confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Volume ratio (current weekly volume / 4-period average)
    vol_ma_4 = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_ma_4_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_4)
    
    # ATR(4) for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = pd.Series(tr_1w).ewm(span=4, adjust=False, min_periods=4).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 7, 4)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_4_aligned[i]) or
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current weekly volume above average
        volume_filter = volume_1w[i] > vol_ma_4_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_1w_aligned[i] > 0.005 * close[i]  # At least 0.5% ATR
        
        # Entry conditions: Weekly R1/S1 breakout with volume and trend
        long_breakout = close[i] > R1_aligned[i]
        short_breakout = close[i] < S1_aligned[i]
        
        long_entry = uptrend and long_breakout and volume_filter and vol_filter
        short_entry = downtrend and short_breakout and volume_filter and vol_filter
        
        # Exit conditions: Weekly S2/R2 retracement (deeper pullback)
        long_exit = close[i] < S2_aligned[i]
        short_exit = close[i] > R2_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0