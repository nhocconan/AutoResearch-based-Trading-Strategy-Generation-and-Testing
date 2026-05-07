#!/usr/bin/env python3
name = "6h_RiverFlow_Trend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # River Flow: 1-day EMA(50) trend
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # River Flow: 1-day volume strength (volume > 20-period average)
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # River Flow: 6-hour volatility filter (ATR-based)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA50 (1d), volume strong, and volatility above average
            price_above_ema = close[i] > ema_50_1d_aligned[i]
            volume_strong = df_1d['volume'].iloc[-1] > vol_ma_20_1d_aligned[i] if len(df_1d) > 0 else False
            vol_conditions = atr[i] > np.nanmean(atr[max(0, i-50):i]) * 0.8
            
            if price_above_ema and volume_strong and vol_conditions:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 (1d), volume strong, and volatility above average
            elif (not price_above_ema) and volume_strong and vol_conditions:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below EMA50 or volatility drops significantly
            if close[i] < ema_50_1d_aligned[i] or atr[i] < np.nanmean(atr[max(0, i-50):i]) * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above EMA50 or volatility drops significantly
            if close[i] > ema_50_1d_aligned[i] or atr[i] < np.nanmean(atr[max(0, i-50):i]) * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: River Flow strategy for 6h timeframe
# - Uses 1-day EMA(50) as primary trend filter (works in both bull/bear markets)
# - Requires 1-day volume strength (>20-period average) for institutional confirmation
# - Uses 6-hour ATR volatility filter to avoid low-volatility whipsaws
# - Long when price > EMA50 + volume strong + volatility adequate
# - Short when price < EMA50 + volume strong + volatility adequate
# - Exits when price crosses EMA50 or volatility drops significantly
# - Position size 0.25 targets 15-35 trades/year, avoiding fee drag
# - River Flow concept: follows institutional money flow like a river follows terrain