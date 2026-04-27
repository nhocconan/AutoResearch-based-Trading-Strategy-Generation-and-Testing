#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    4h Williams %R with 1d Trend Filter and Volume Spike.
    Long: Williams %R < -80 (oversold) + price > 1d EMA50 + volume spike
    Short: Williams %R > -20 (overbought) + price < 1d EMA50 + volume spike
    Exit: Williams %R crosses -50 (mean reversion) or trailing stop
    Designed for mean reversion in ranging markets with trend filter for bias.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams %R calculation and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.full(len(close_1d), np.nan)
    lookback = 14
    for i in range(lookback - 1, len(close_1d)):
        highest_high = np.max(high_1d[i-lookback+1:i+1])
        lowest_low = np.min(low_1d[i-lookback+1:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50  # EMA50
    
    # Align daily indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h ATR(14) for trailing stop
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, vol_period, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above EMA50 + volume spike
            if williams_r_aligned[i] < -80 and price > ema_50_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) + price below EMA50 + volume spike
            elif williams_r_aligned[i] > -20 and price < ema_50_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion) or trailing stop
            if williams_r_aligned[i] > -50 or price < ema_50_1d_aligned[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion) or trailing stop
            if williams_r_aligned[i] < -50 or price > ema_50_1d_aligned[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_14_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0