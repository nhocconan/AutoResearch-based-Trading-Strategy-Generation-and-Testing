# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day volume-weighted average price (VWAP) deviation with 1-day ADX trend filter.
# Long when price closes below 1-day VWAP by 1.5 standard deviations AND 1-day ADX > 25 (trending market).
# Short when price closes above 1-day VWAP by 1.5 standard deviations AND 1-day ADX > 25.
# Exit when price returns to within 0.5 standard deviations of 1-day VWAP.
# Uses mean-reversion in trending markets with proper trend filter to avoid whipsaws.
# Target: 20-40 trades per year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day typical price and VWAP components
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    tpv_1d = typical_price_1d * volume_1d
    
    # Calculate cumulative sums for VWAP (reset daily)
    cum_tpv = np.cumsum(tpv_1d)
    cum_vol = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_tpv, cum_vol, out=np.full_like(cum_tpv, np.nan), where=cum_vol!=0)
    
    # Reset VWAP at day start (where volume resets)
    vol_reset = np.concatenate(([True], volume_1d[1:] < volume_1d[:-1]))
    vwap_1d = np.where(vol_reset, typical_price_1d, vwap_1d)
    
    # Calculate rolling standard deviation of price deviation from VWAP (20-period)
    price_dev_1d = close_1d - vwap_1d
    std_dev_20 = np.full(len(close_1d), np.nan)
    for i in range(19, len(close_1d)):
        std_dev_20[i] = np.std(price_dev_1d[i-19:i+1])
    
    # Calculate 1-day ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate(([close[0]], close[:-1])))
        tr3 = np.abs(low - np.concatenate(([close[0]], close[:-1])))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.concatenate(([high[0]], high[:-1]))) > 
                          (np.concatenate(([low[0]], low[:-1])) - low), 
                          np.maximum(high - np.concatenate(([high[0]], high[:-1])), 0), 0)
        dm_minus = np.where((np.concatenate(([low[0]], low[:-1])) - low) > 
                           (high - np.concatenate(([high[0]], high[:-1]))), 
                          np.maximum(np.concatenate(([low[0]], low[:-1])) - low, 0), 0)
        
        # Smoothed values
        atr = np.full(len(high), np.nan)
        dm_plus_smooth = np.full(len(high), np.nan)
        dm_minus_smooth = np.full(len(high), np.nan)
        
        if len(high) >= period:
            # Initial values
            atr[period-1] = np.mean(tr[:period])
            dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
            dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
            
            # Wilder smoothing
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = np.full(len(high), np.nan)
        minus_di = np.full(len(high), np.nan)
        dx = np.full(len(high), np.nan)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
                minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX (smoothed DX)
        adx = np.full(len(high), np.nan)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    std_dev_20_aligned = align_htf_to_ltf(prices, df_1d, std_dev_20)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h VWAP for entry/exit (using same method as daily but on 4h data)
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv_4h = np.cumsum(tpv)
    cum_vol_4h = np.cumsum(volume)
    vwap_4h = np.divide(cum_tpv_4h, cum_vol_4h, out=np.full_like(cum_tpv_4h, np.nan), where=cum_vol_4h!=0)
    
    # Reset 4h VWAP at 4h boundaries (simplified: every 16th bar for 15m->4h, but we use actual 4h alignment)
    # Since we're on 4h timeframe, each bar is a 4h bar, so VWAP resets daily
    # We'll use the 1-day VWAP aligned for consistency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d VWAP, std dev, and ADX
    start_idx = max(30, 19)  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(std_dev_20_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        std_dev = std_dev_20_aligned[i]
        adx = adx_1d_aligned[i]
        
        # Calculate deviation from VWAP in standard deviations
        if std_dev > 0:
            dev_stdev = (price - vwap) / std_dev
        else:
            dev_stdev = 0
        
        # Trend filter: only trade in trending markets (ADX > 25)
        trend_filter = adx > 25
        
        if position == 0:
            # Long: price significantly below VWAP in trending market
            if dev_stdev < -1.5 and trend_filter:
                signals[i] = size
                position = 1
            # Short: price significantly above VWAP in trending market
            elif dev_stdev > 1.5 and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns near VWAP
            if abs(dev_stdev) < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns near VWAP
            if abs(dev_stdev) < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_VWAPDeviation_ADXTrend_MeanReversion"
timeframe = "4h"
leverage = 1.0