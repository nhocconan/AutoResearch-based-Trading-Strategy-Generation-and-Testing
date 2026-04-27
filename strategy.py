#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX trend + 12h VWAP mean reversion with volume confirmation
# ADX identifies trending markets (ADX > 25), VWAP acts as dynamic support/resistance.
# Long when price > VWAP in uptrend, short when price < VWAP in downtrend.
# Volume filter ensures momentum behind moves. Works in bull/bear by following trend.
# Target: 60-120 total trades over 4 years (~15-30/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate VWAP for each 12h bar (cumulative within day, reset at new day)
    vwap_12h = np.full(len(df_12h), np.nan)
    for i in range(len(df_12h)):
        # Typical price * cumulative volume / cumulative volume
        typ_price = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        if i == 0:
            vwap_12h[i] = typ_price
        else:
            # VWAP = (prev VWAP * prev vol + typ price * vol) / (prev vol + vol)
            # Simplified: cumulative TP*V / cumulative V
            cum_vol = np.sum(volume_12h[:i+1])
            if cum_vol > 0:
                cum_tpv = np.sum(((high_12h[:i+1] + low_12h[:i+1] + close_12h[:i+1]) / 3.0) * volume_12h[:i+1])
                vwap_12h[i] = cum_tpv / cum_vol
    
    # Align VWAP to 4h timeframe (wait for 12h close)
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # 12h ADX trend filter (14-period)
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros(len(df_12h))
    minus_dm = np.zeros(len(df_12h))
    tr = np.zeros(len(df_12h))
    
    for i in range(1, len(df_12h)):
        high_diff = high_12h[i] - high_12h[i-1]
        low_diff = low_12h[i-1] - low_12h[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high_12h[i] - low_12h[i], 
                    abs(high_12h[i] - close_12h[i-1]),
                    abs(low_12h[i] - close_12h[i-1]))
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: prev * (period-1)/period + current/period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    plus_di = 100 * wilders_smoothing(plus_dm, period) / wilders_smoothing(tr, period)
    minus_di = 100 * wilders_smoothing(minus_dm, period) / wilders_smoothing(tr, period)
    dx = np.zeros(len(df_12h))
    dx[:] = np.where((plus_di + minus_di) != 0, 
                     100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_12h = wilders_smoothing(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h VWAP (1 bar), ADX (14+14=28), volume MA (20)
    start_idx = max(1, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: sufficient volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h ADX
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price above VWAP in uptrend with volume
            if price > vwap_aligned[i] and trending and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below VWAP in downtrend with volume
            elif price < vwap_aligned[i] and trending and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below VWAP or trend weakens
            if price <= vwap_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above VWAP or trend weakens
            if price >= vwap_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ADX_Trend_VWAP_MeanReversion_12h"
timeframe = "4h"
leverage = 1.0