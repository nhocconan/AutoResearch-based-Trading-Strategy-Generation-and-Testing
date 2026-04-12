#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly EMA(20) for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR(10) for volatility filter
    tr1_w = np.abs(high_1w - low_1w)
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_w[0] = tr2_w[0] = tr3_w[0] = np.nan
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(10, len(df_1w)):
        atr_1w[i] = np.mean(tr_w[i-10:i+1])
    
    # Calculate daily EMA(50) for intermediate trend
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1_d = np.abs(high_1d - low_1d)
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_d[0] = tr2_d[0] = tr3_d[0] = np.nan
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_d[i-14:i+1])
    
    # Calculate daily volume moving average
    vol_s_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to 6h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Align daily indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5 * 20-period daily MA (scaled)
        # Scale daily volume MA to 6h by dividing by 4 (approximate)
        vol_filter = volume[i] > 1.5 * (vol_ma_20_1d_aligned[i] / 4)
        
        # Volatility filter: weekly ATR > 0.8 * its 20-period MA (avoid low volatility)
        atr_ma_20_1w = np.full(len(df_1w), np.nan)
        for j in range(30, len(df_1w)):  # 10 + 19 for 20-period MA
            if not np.isnan(np.mean(atr_1w[j-19:j+1])):
                atr_ma_20_1w[j] = np.mean(atr_1w[j-19:j+1])
        atr_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_20_1w)
        vol_filter_weekly = (not np.isnan(atr_ma_20_1w_aligned[i]) and 
                           atr_1w_aligned[i] > 0.8 * atr_ma_20_1w_aligned[i])
        
        # Trend alignment: weekly EMA20 direction + price above/below daily EMA50
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        daily_uptrend = close[i] > ema_50_1d_aligned[i]
        daily_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: aligned trends + volume + volatility filter
        long_entry = weekly_uptrend and daily_uptrend and vol_filter and vol_filter_weekly
        short_entry = weekly_downtrend and daily_downtrend and vol_filter and vol_filter_weekly
        
        # Exit conditions: trend misalignment
        long_exit = not (weekly_uptrend and daily_uptrend)
        short_exit = not (weekly_downtrend and daily_downtrend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_ema_trend_vol_filter_v1"
timeframe = "6h"
leverage = 1.0