#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Vortex Indicator + 1d Volume Spike + Trend Filter
# Vortex detects trend direction by comparing upward/downward movement.
# Combined with 1d volume spike to confirm institutional interest and 1d EMA for trend filter.
# Works in bull/bear by filtering Vortex signal with 1d EMA trend.
# Target: 20-60 total trades over 4 years (~5-15/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Vortex components for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])  # Upward movement
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])  # Downward movement
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Vortex Indicator (14-period)
    period = 14
    tr_sum = np.full(len(df_1d), np.nan)
    vm_plus_sum = np.full(len(df_1d), np.nan)
    vm_minus_sum = np.full(len(df_1d), np.nan)
    
    for i in range(period, len(df_1d)):
        tr_sum[i] = np.sum(tr[i-period+1:i+1])
        vm_plus_sum[i] = np.sum(vm_plus[i-period+1:i+1])
        vm_minus_sum[i] = np.sum(vm_minus[i-period+1:i+1])
    
    vi_plus = np.divide(vm_plus_sum, tr_sum, out=np.full_like(tr_sum, np.nan), where=tr_sum!=0)
    vi_minus = np.divide(vm_minus_sum, tr_sum, out=np.full_like(tr_sum, np.nan), where=tr_sum!=0)
    
    # Align Vortex indicators to 4h timeframe (wait for 1d close)
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.8 x 24-period average (4 days of 4h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (1 bar), Vortex (14), EMA (34), volume MA (24)
    start_idx = max(1, 14, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.8 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        if position == 0:
            # Long: VI+ > VI- with volume and bullish trend
            if vi_plus_aligned[i] > vi_minus_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: VI- > VI+ with volume and bearish trend
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: VI- > VI+ (trend change) or trend turns bearish
            if vi_minus_aligned[i] > vi_plus_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: VI+ > VI- (trend change) or trend turns bullish
            if vi_plus_aligned[i] > vi_minus_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Vortex_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0