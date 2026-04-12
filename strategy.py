#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Trend_v3
Hypothesis: 4h close above/below daily Camarilla R4/S4 levels with 12h EMA(21) trend filter and volume confirmation.
Improvements over v2: Added minimum holding period (3 bars) to reduce churn, tightened volume threshold to 2.5x,
and added ADX(14) filter on 4h to avoid choppy markets. Target: 25-40 trades/year for better generalization.
Works in bull/bear via EMA trend filter and mean-reversion exit at daily pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Breakout_Trend_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Daily Camarilla levels (R4/S4 for stronger breakouts)
    r4_1d = close_1d + range_1d * 1.5
    s4_1d = close_1d - range_1d * 1.5
    
    # === 12H EMA(21) FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    if len(close_12h) >= 21:
        ema_21_12h = np.zeros_like(close_12h)
        ema_21_12h[0] = close_12h[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_12h)):
            ema_21_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_21_12h[i-1]
    else:
        ema_21_12h = np.full_like(close_12h, np.nan)
    
    # === 4H ADX(14) FOR TREND STRENGTH ===
    # Calculate +DI, -DI, DX
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # Align daily and 12h data to 4h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    adx_aligned = align_htf_to_ltf(prices, pd.DataFrame({'index': range(len(adx))}), adx)  # dummy df for alignment
    
    # Volume average (20-period for 4h = ~1.3 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    for i in range(60, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            bars_since_entry += 1
            continue
        
        bars_since_entry += 1
        
        # Volume confirmation: at least 2.5x average (tighter than v2)
        vol_confirm = volume[i] > 2.5 * vol_avg[i]
        
        # Trend filter: price above/below 12h EMA(21)
        price_above_ema = close[i] > ema_21_12h_aligned[i]
        price_below_ema = close[i] < ema_21_12h_aligned[i]
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Breakout entries at daily S4/R4 with volume, trend, and ADX filters
        long_setup = (close[i] > r4_1d_aligned[i]) and vol_confirm and price_above_ema and strong_trend
        short_setup = (close[i] < s4_1d_aligned[i]) and vol_confirm and price_below_ema and strong_trend
        
        # Exit when price returns to daily pivot (mean reversion) OR after minimum hold
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        min_hold_exit = bars_since_entry >= 3  # minimum 3 bars holding
        
        if long_setup and position != 1:
            position = 1
            bars_since_entry = 0
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            bars_since_entry = 0
            signals[i] = -0.25
        elif (exit_long and position == 1) or (exit_short and position == -1) or min_hold_exit:
            position = 0
            bars_since_entry = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals