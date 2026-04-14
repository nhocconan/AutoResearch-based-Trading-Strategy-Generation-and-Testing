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
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ADX (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        tr = np.zeros(len(high))
        dm_plus = np.zeros(len(high))
        dm_minus = np.zeros(len(high))
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            dm_plus[i] = max(high[i] - high[i-1], 0)
            dm_minus[i] = max(low[i-1] - low[i], 0)
            dm_plus[i] = dm_plus[i] if dm_plus[i] > dm_minus[i] else 0
            dm_minus[i] = dm_minus[i] if dm_minus[i] > dm_plus[i] else 0
        
        # Smooth TR, DM+
        tr_smooth = np.zeros(len(tr))
        dm_plus_smooth = np.zeros(len(dm_plus))
        dm_minus_smooth = np.zeros(len(dm_minus))
        
        if len(tr) >= period:
            tr_smooth[period-1] = np.sum(tr[:period])
            dm_plus_smooth[period-1] = np.sum(dm_plus[:period])
            dm_minus_smooth[period-1] = np.sum(dm_minus[:period])
            
            for i in range(period, len(tr)):
                tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
                dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
                dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # Calculate DI and DX
        di_plus = np.zeros(len(tr))
        di_minus = np.zeros(len(tr))
        dx = np.zeros(len(tr))
        
        for i in range(period-1, len(tr)):
            if tr_smooth[i] != 0:
                di_plus[i] = 100 * (dm_plus_smooth[i] / tr_smooth[i])
                di_minus[i] = 100 * (dm_minus_smooth[i] / tr_smooth[i])
                if (di_plus[i] + di_minus[i]) != 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # Calculate ADX (smoothed DX)
        adx = np.zeros(len(dx))
        if len(dx) >= 2 * period - 1:
            adx[2*period-2] = np.sum(dx[period-1:2*period-1]) / period
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Weekly 200 EMA for trend filter
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        multiplier = 2 / (200 + 1)
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * multiplier) + (ema_200_1w[i-1] * (1 - multiplier))
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily close for price action
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1w_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when weekly ADX > 25 (trending market)
        if adx_1w_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly 200 EMA and weekly ADX rising
            if close[i] > ema_200_1w_aligned[i] and adx_1w_aligned[i] > adx_1w_aligned[i-1]:
                position = 1
                signals[i] = position_size
            # Short: Price below weekly 200 EMA and weekly ADX rising
            elif close[i] < ema_200_1w_aligned[i] and adx_1w_aligned[i] > adx_1w_aligned[i-1]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below weekly 200 EMA or ADX drops below 20
            if close[i] < ema_200_1w_aligned[i] or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above weekly 200 EMA or ADX drops below 20
            if close[i] > ema_200_1w_aligned[i] or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_ADX200EMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0