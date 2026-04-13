#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using weekly Camarilla pivot levels (S3/R3 for mean reversion, S4/R4 for breakout)
    # with 1d ADX trend filter and volume confirmation. In ranging markets (ADX<25), fade extremes.
    # In trending markets (ADX>25), breakout continuation. Weekly pivots provide strong structure.
    # Discrete sizing (0.25) minimizes fee drag. Target: 12-37 trades/year for 6h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1w data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data for ADX trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate weekly Camarilla levels (based on previous week)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S3 = C - (Range * 1.1000/2)
    # S4 = C - (Range * 1.1000)
    # R3 = C + (Range * 1.1000/2)
    # R4 = C + (Range * 1.1000)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    s3_1w = close_1w - (range_1w * 1.1000 / 2.0)
    s4_1w = close_1w - (range_1w * 1.1000)
    r3_1w = close_1w + (range_1w * 1.1000 / 2.0)
    r4_1w = close_1w + (range_1w * 1.1000)
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(high))
        minus_di = 100 * (np.zeros_like(high))
        dx = 100 * (np.zeros_like(high))
        
        plus_sm = np.zeros_like(high)
        minus_sm = np.zeros_like(high)
        plus_sm[period] = np.sum(plus_dm[1:period+1])
        minus_sm[period] = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            plus_sm[i] = (plus_sm[i-1] * (period-1) + plus_dm[i]) / period
            minus_sm[i] = (minus_sm[i-1] * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_sm[i] / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_sm[i] / atr[i] if atr[i] != 0 else 0
            dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx = np.zeros_like(dx)
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Get 1d volume average (20-period)
    vol_avg_20_1d = np.zeros_like(volume_1d)
    for i in range(20, len(volume_1d)):
        vol_avg_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align all HTF indicators to 6h primary timeframe
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average (using 1d volume as proxy)
        # Since we don't have 6h volume in 1d data, use 1d volume of current day
        idx_1d = i // 4  # 6h bars in 1d timeframe (4 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend regime: ADX > 25 = trending, ADX < 25 = ranging
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25
        is_ranging = adx_value <= 25
        
        # Entry conditions based on regime
        # In ranging market: fade at S3/R3 (mean reversion)
        # In trending market: breakout at S4/R4 (continuation)
        enter_long = False
        enter_short = False
        
        if is_ranging and volume_confirmed:
            # Fade at extremes in ranging market
            enter_long = close[i] <= s3_1w_aligned[i]
            enter_short = close[i] >= r3_1w_aligned[i]
        elif is_trending and volume_confirmed:
            # Breakout continuation in trending market
            enter_long = close[i] >= r4_1w_aligned[i]
            enter_short = close[i] <= s4_1w_aligned[i]
        
        # Stoploss: 1.5x ATR based on 1d true range (simplified using daily range)
        daily_range = high_1d[idx_1d] - low_1d[idx_1d] if idx_1d < len(high_1d) else 0
        stop_distance = daily_range * 0.75  # 75% of daily range
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1w_1d_camarilla_pivot_adx_volume_v1"
timeframe = "6h"
leverage = 1.0