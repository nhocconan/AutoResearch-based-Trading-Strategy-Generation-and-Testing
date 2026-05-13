#!/usr/bin/env python3
"""
6h_ADX_Ichimoku_Cloud_Breakout_Trend
Hypothesis: Combine ADX trend strength with Ichimoku cloud breakout on 6h timeframe, filtered by weekly trend for bias.
In trending markets (ADX>25), price breaking above/below Ichimoku cloud with TK cross signals continuation.
Weekly trend filter avoids counter-trend trades. Works in bull/bear by only taking trades aligned with higher timeframe trend.
Target: 15-30 trades/year per symbol.
"""

name = "6h_ADX_Ichimoku_Cloud_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smooth TR, DM+ and DM-
        tr_smooth = np.zeros_like(tr)
        dm_plus_smooth = np.zeros_like(tr)
        dm_minus_smooth = np.zeros_like(tr)
        
        tr_smooth[period] = np.sum(tr[1:period+1])
        dm_plus_smooth[period] = np.sum(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.sum(dm_minus[1:period+1])
        
        for i in range(period+1, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # Avoid division by zero
        dm_plus_smooth = np.where(tr_smooth == 0, 0, dm_plus_smooth)
        dm_minus_smooth = np.where(tr_smooth == 0, 0, dm_minus_smooth)
        
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        if len(dx) >= 2*period:
            adx[2*period] = np.sum(dx[period:2*period+1]) / period
            for i in range(2*period+1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    trending = adx > 25
    
    # Ichimoku Cloud calculation (9, 26, 52)
    def calculate_ichimoku(high, low, close):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = np.zeros_like(high)
        period9_low = np.zeros_like(low)
        for i in range(len(high)):
            if i >= 8:
                period9_high[i] = np.max(high[i-8:i+1])
                period9_low[i] = np.min(low[i-8:i+1])
            else:
                period9_high[i] = np.nan
                period9_low[i] = np.nan
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = np.zeros_like(high)
        period26_low = np.zeros_like(low)
        for i in range(len(high)):
            if i >= 25:
                period26_high[i] = np.max(high[i-25:i+1])
                period26_low[i] = np.min(low[i-25:i+1])
            else:
                period26_high[i] = np.nan
                period26_low[i] = np.nan
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = np.zeros_like(high)
        period52_low = np.zeros_like(low)
        for i in range(len(high)):
            if i >= 51:
                period52_high[i] = np.max(high[i-51:i+1])
                period52_low[i] = np.min(low[i-51:i+1])
            else:
                period52_high[i] = np.nan
                period52_low[i] = np.nan
        senkou_b = ((period52_high + period52_low) / 2)
        
        # Current cloud boundaries (shifted back by 26 periods to align with current price)
        senkou_a_shifted = np.roll(senkou_a, 26)
        senkou_b_shifted = np.roll(senkou_b, 26)
        senkou_a_shifted[:26] = np.nan
        senkou_b_shifted[:26] = np.nan
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
        cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
        
        return tenkan, kijun, cloud_top, cloud_bottom
    
    tenkan, kijun, cloud_top, cloud_bottom = calculate_ichimoku(high, low, close)
    
    # TK Cross signals
    tk_cross_up = np.where((tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1)), 1, 0)
    tk_cross_down = np.where((tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1)), 1, 0)
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Weekly trend filter (Higher Timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from sufficient lookback for all indicators
    start_idx = max(52, 26)  # Ichimoku needs 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any key value is NaN
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            signals[i] = 0.0
            continue
            
        # Get current values
        adx_val = adx[i]
        trending_now = trending[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        tk_up = tk_cross_up[i]
        tk_down = tk_cross_down[i]
        weekly_up = uptrend_1w_aligned[i]
        weekly_down = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: ADX trending, price above cloud, TK cross up, weekly uptrend
            if trending_now and price_above and tk_up and weekly_up:
                signals[i] = 0.25
                position = 1
            # SHORT: ADX trending, price below cloud, TK cross down, weekly downtrend
            elif trending_now and price_below and tk_down and weekly_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below cloud or TK cross down or weekly trend turns down
            if price_below or tk_down or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above cloud or TK cross up or weekly trend turns up
            if price_above or tk_up or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals