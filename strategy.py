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
    
    # Get weekly data for calculations (called ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly high and low (for trend bias)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly EMA (10-period) for trend filter
    ema_10_weekly = np.full(len(high_weekly), np.nan)
    if len(high_weekly) >= 10:
        alpha = 2 / (10 + 1)
        ema_10_weekly[0] = high_weekly[0]  # Use high for bullish bias
        for i in range(1, len(high_weekly)):
            ema_10_weekly[i] = alpha * high_weekly[i] + (1 - alpha) * ema_10_weekly[i-1]
    
    # Weekly EMA (10-period) using low for bearish bias
    ema_10_weekly_low = np.full(len(low_weekly), np.nan)
    if len(low_weekly) >= 10:
        alpha = 2 / (10 + 1)
        ema_10_weekly_low[0] = low_weekly[0]
        for i in range(1, len(low_weekly)):
            ema_10_weekly_low[i] = alpha * low_weekly[i] + (1 - alpha) * ema_10_weekly_low[i-1]
    
    # Daily ATR (14-period) for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    close_daily_prev = np.roll(close_daily, 1)
    close_daily_prev[0] = close_daily[0]
    
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - close_daily_prev)
    tr3 = np.abs(low_daily - close_daily_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_daily = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_daily[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_daily[i] = (atr_14_daily[i-1] * 13 + tr[i]) / 14
    
    # Align weekly indicators to 6h timeframe
    ema_10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_10_weekly)
    ema_10_weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, ema_10_weekly_low)
    atr_14_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_14_daily)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(10, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_weekly_aligned[i]) or np.isnan(ema_10_weekly_low_aligned[i]) or 
            np.isnan(atr_14_daily_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.8x average volume
        vol_filter = vol_ratio > 1.8
        
        if position == 0:
            # Long: Price above weekly EMA(10) high with volume and ATR filter
            if price > ema_10_weekly_aligned[i] and vol_filter and atr_14_daily_aligned[i] > 0:
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA(10) low with volume and ATR filter
            elif price < ema_10_weekly_low_aligned[i] and vol_filter and atr_14_daily_aligned[i] > 0:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below weekly EMA(10) low or volatility spike
            if price < ema_10_weekly_low_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above weekly EMA(10) high or volatility spike
            if price > ema_10_weekly_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyEMA10_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0