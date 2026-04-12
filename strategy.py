#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Volume_Regime_v2"
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
    
    # === 1W DATA FOR TREND (EMA 13) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 1W VOLUME FOR CONFIRMATION ===
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 1D DATA FOR CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Map each 12h bar to previous day's OHLC
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    pivots_close = np.full(n, np.nan)
    
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        prev_date = current_time.date() - pd.Timedelta(days=1)
        
        # Find previous day in daily data
        for j in range(len(df_1d)):
            if pd.Timestamp(df_1d.iloc[j]['open_time']).date() == prev_date:
                pivots_high[i] = high_1d[j]
                pivots_low[i] = low_1d[j]
                pivots_close[i] = close_1d[j]
                break
    
    # Calculate Camarilla H4 and L4 levels (stronger breakout levels)
    H4 = pivots_close + (pivots_high - pivots_low) * 1.1 / 2
    L4 = pivots_close - (pivots_high - pivots_low) * 1.1 / 2
    
    # === 12H DATA FOR CHOPPINESS INDEX ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for Chop Index
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Sum of true range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Chop Index: 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum / (atr_12h * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)  # neutral when invalid
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or np.isnan(H4[i]) or 
            np.isnan(L4[i]) or np.isnan(pivots_close[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend: price above/below weekly EMA
        above_week_ema = close[i] > ema_1w_aligned[i]
        below_week_ema = close[i] < ema_1w_aligned[i]
        
        # Chop regime: Chop > 50 = ranging (mean revert), Chop < 50 = trending
        chop_low = chop_aligned[i] < 50
        
        # Volume confirmation: current volume > weekly average
        strong_volume = volume[i] > vol_ma_1w_aligned[i]
        
        # Long: price breaks above H4 in trending/up market with volume
        long_signal = (close[i] > H4[i] and 
                      above_week_ema and 
                      chop_low and 
                      strong_volume)
        
        # Short: price breaks below L4 in trending/down market with volume
        short_signal = (close[i] < L4[i] and 
                       below_week_ema and 
                       chop_low and 
                       strong_volume)
        
        # Exit: chop increases (range) or price returns to pivot
        exit_long = (position == 1 and 
                    (chop_aligned[i] > 60 or close[i] < pivots_close[i]))
        exit_short = (position == -1 and 
                     (chop_aligned[i] > 60 or close[i] > pivots_close[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals