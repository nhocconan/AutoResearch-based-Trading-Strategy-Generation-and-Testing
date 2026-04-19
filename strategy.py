#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSIOscillator_TrendFollow_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using prior week's data (5 days ago approximation)
    if len(high_1d) < 5:
        return np.zeros(n)
    
    # Use data from 5 days ago to represent prior week
    high_5d_ago = np.roll(high_1d, 5)
    low_5d_ago = np.roll(low_1d, 5)
    close_5d_ago = np.roll(close_1d, 5)
    
    # Set first 5 values to NaN (no prior week data)
    high_5d_ago[:5] = np.nan
    low_5d_ago[:5] = np.nan
    close_5d_ago[:5] = np.nan
    
    # Weekly pivot point: P = (H + L + C) / 3
    weekly_pivot = (high_5d_ago + low_5d_ago + close_5d_ago) / 3.0
    
    # Weekly R1 and S1: R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - low_5d_ago
    weekly_s1 = 2 * weekly_pivot - high_5d_ago
    
    # Align weekly pivot levels to 4h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 12-period ATR for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or \
           np.isnan(ema20_1w_aligned[i]) or np.isnan(atr_12[i]) or np.isnan(vol_ma_20[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_12[i]
        ema_trend = ema20_1w_aligned[i]
        rsi_val = rsi[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        rsi_bullish = rsi_val > 50
        rsi_bearish = rsi_val < 50
        
        if position == 0:
            # Long: price breaks above weekly R1 + uptrend + volume + RSI bullish
            if price > r1 and price > ema_trend and volume_confirmed and rsi_bullish:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + downtrend + volume + RSI bearish
            elif price < s1 and price < ema_trend and volume_confirmed and rsi_bearish:
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