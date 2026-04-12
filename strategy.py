#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and daily for ATR/volume
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 20 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly close for EMA trend
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema_vals = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        ema_1w[:len(ema_vals)] = ema_vals
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = np.nanmean(tr[i-13:i+1])
    
    # Daily volume average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align to 12h timeframe
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1d_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(80, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_12h[i]) or 
            np.isnan(atr_1d_12h[i]) or np.isnan(vol_ma_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below 21 EMA
        trend_up = close[i] > ema_1w_12h[i]
        trend_down = close[i] < ema_1w_12h[i]
        
        # Volatility filter: current ATR > 1.5x 20-day average ATR
        vol_filter = atr_1d_12h[i] > (np.nanmean(atr_1d_12h[max(0,i-20):i]) * 1.5) if i >= 20 else False
        
        # Volume filter: current volume > 1.5x 20-day average volume
        vol_ma_val = vol_ma_1d_12h[i]
        vol_filter = vol_filter and (volume[i] > vol_ma_val * 1.5) if not np.isnan(vol_ma_val) else False
        
        # Entry conditions: break of weekly EMA with volume/vol confirmation
        long_entry = trend_up and vol_filter
        short_entry = trend_down and vol_filter
        
        # Exit conditions: opposite EMA cross or volatility drop
        long_exit = not trend_up or not vol_filter
        short_exit = not trend_down or not vol_filter
        
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

name = "12h_1w_ema_trend_vol_vol_filter_v1"
timeframe = "12h"
leverage = 1.0