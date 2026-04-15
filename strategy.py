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
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop (primary trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly EMA(34) for trend direction
    weekly_ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_34)
    
    # Get daily HTF data once before loop (HTF pivot levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily pivot points (Camarilla style)
    # P = (H + L + C) / 3
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6
    # S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2
    # S4 = C - (H-L)*1.1/2
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_hl = daily_high - daily_low
    r1 = pivot + range_hl * 1.1 / 12
    s1 = pivot - range_hl * 1.1 / 12
    r2 = pivot + range_hl * 1.1 / 6
    s2 = pivot - range_hl * 1.1 / 6
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    r4 = pivot + range_hl * 1.1 / 2
    s4 = pivot - range_hl * 1.1 / 2
    
    # Align HTF indicators to 6h timeframe with proper delay
    weekly_ema_34_6h = weekly_ema_34_aligned  # already aligned
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_34_6h[i]) or np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA(34)
        # Long when price > weekly EMA, short when price < weekly EMA
        trend_long = close[i] > weekly_ema_34_6h[i]
        trend_short = close[i] < weekly_ema_34_6h[i]
        
        # Entry conditions:
        # Long: price breaks above R3 with volume confirmation AND in weekly uptrend
        # Short: price breaks below S3 with volume confirmation AND in weekly downtrend
        # Volume: > 1.5x average
        # Volatility: ATR > 0.5% of price (avoid low volatility chop)
        # Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above R3 (strong resistance)
        if (trend_long and
            close[i] > r3_6h[i] and            # 6h price above R3 (strong breakout)
            volume_ratio[i] > 1.5 and          # Strong volume confirmation
            atr_14[i] > 0.005 * close[i]):     # Adequate volatility
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below S3 (strong support)
        elif (trend_short and
              close[i] < s3_6h[i] and          # 6h price below S3 (strong breakdown)
              volume_ratio[i] > 1.5 and        # Strong volume confirmation
              atr_14[i] > 0.005 * close[i]):   # Adequate volatility
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA34_CamarillaR3S3_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0