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
    
    # Get weekly data once for HTF context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA(21) for trend
    ema_21_w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly ATR(14) for volatility
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to daily timeframe
    ema_21_w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_w)
    atr_14_w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_w)
    
    # Daily EMA(50) for additional trend confirmation
    ema_50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily ATR(14) for volatility-based sizing
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_14_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_w_aligned[i]) or np.isnan(atr_14_w_aligned[i]) or 
            np.isnan(ema_50_d[i]) or np.isnan(atr_14_d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above weekly EMA21 and daily EMA50
        trend_up = close[i] > ema_21_w_aligned[i] and close[i] > ema_50_d[i]
        trend_down = close[i] < ema_21_w_aligned[i] and close[i] < ema_50_d[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_14_d[i] < 2.0 * atr_14_w_aligned[i]
        
        # Volume filter: above average volume
        vol_filter_volume = volume[i] > vol_ma[i]
        
        # Entry conditions - selective to reduce trades
        long_entry = trend_up and vol_filter and vol_filter_volume
        short_entry = trend_down and vol_filter and vol_filter_volume
        
        # Exit conditions: opposite trend or volatility spike
        long_exit = not trend_up or (atr_14_d[i] > 2.5 * atr_14_w_aligned[i])
        short_exit = not trend_down or (atr_14_d[i] > 2.5 * atr_14_w_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "1d_WeeklyEMA21_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0