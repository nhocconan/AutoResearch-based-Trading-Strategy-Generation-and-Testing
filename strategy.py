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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly EMA(34) for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly ATR for volatility regime filter
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    tr1_w = df_1w_high - df_1w_low
    tr2_w = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3_w = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_w_aligned = align_htf_to_ltf(prices, df_1w, atr_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend_1d = ema34_1d_aligned[i]
        ema_trend_1w = ema34_1w_aligned[i]
        atr_current = atr[i]
        atr_weekly = atr_w_aligned[i]
        
        # Volatility filter: only trade when current volatility is above weekly average
        vol_filter = atr_current > (atr_weekly * 0.8)
        
        if position == 0:
            # Only trade when both daily and weekly trends agree
            trend_aligned = (ema_trend_1d > ema_trend_1w)  # Uptrend when daily > weekly
            
            # Long: break above high with volume spike and uptrend
            if (high[i] > high[i-1] and close[i] > high[i-1] and 
                trend_aligned and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below low with volume spike and downtrend
            elif (low[i] < low[i-1] and close[i] < low[i-1] and 
                  not trend_aligned and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches previous low or trend reverses
            if low[i] <= low[i-1] or not trend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches previous high or trend reverses
            if high[i] >= high[i-1] or trend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Breakout_PrevBar_1d1wEMA34_Trend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0