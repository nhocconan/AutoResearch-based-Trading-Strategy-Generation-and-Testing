#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Regime_v3
Hypothesis: Refined version focusing on high-probability setups by tightening entry conditions:
- Only trade when price breaks above H3 (long) or below L3 (short) with volume > 2x average
- Require 1d trend filter (price above/below 50 EMA) for directional bias
- Exit at opposite H3/L3 level for mean reversion
- Use volatility regime filter (ADX < 25) to avoid choppy markets
Target: 20-35 trades/year per symbol with focus on quality over quantity.
Works in bull/bear via 1d EMA trend filter and volatility regime avoidance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_Regime_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR INDICATORS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === DAILY EMA50 FOR TREND FILTER ===
    close_series = pd.Series(close_1d)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === DAILY CAMARILLA LEVELS (based on previous day) ===
    # Use previous day's data to avoid look-ahead
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    # Set first value to avoid NaN propagation
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    range_prev = high_prev - low_prev
    
    h3 = close_prev + (range_prev * 1.1 / 4)
    l3 = close_prev - (range_prev * 1.1 / 4)
    h4 = close_prev + (range_prev * 1.1)
    l4 = close_prev - (range_prev * 1.1)
    
    # === DAILY VOLATILITY REGIME: ADX < 25 (low trend strength = avoid chop) ===
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Wilder's smoothing for TR and DM
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(tr, np.nan)
    minus_dm_smooth = np.full_like(tr, np.nan)
    
    for i in range(len(tr)):
        if i < 1:
            continue
        if i < 14:
            if i == 1:
                atr[i] = np.nanmean(tr[1:i+1]) if not np.all(np.isnan(tr[1:i+1])) else np.nan
                plus_dm_smooth[i] = np.nanmean(plus_dm[:i+1]) if i < len(plus_dm) else np.nan
                minus_dm_smooth[i] = np.nanmean(minus_dm[:i+1]) if i < len(minus_dm) else np.nan
            else:
                prev_atr = atr[i-1] if not np.isnan(atr[i-1]) else 0
                prev_plus = plus_dm_smooth[i-1] if not np.isnan(plus_dm_smooth[i-1]) else 0
                prev_minus = minus_dm_smooth[i-1] if not np.isnan(minus_dm_smooth[i-1]) else 0
                atr[i] = (prev_atr * 13 + tr[i]) / 14 if not np.isnan(prev_atr) else np.nan
                plus_dm_smooth[i] = (prev_plus * 13 + (plus_dm[i] if i < len(plus_dm) else 0)) / 14
                minus_dm_smooth[i] = (prev_minus * 13 + (minus_dm[i] if i < len(minus_dm) else 0)) / 14
        else:
            prev_atr = atr[i-1]
            prev_plus = plus_dm_smooth[i-1]
            prev_minus = minus_dm_smooth[i-1]
            atr[i] = (prev_atr * 13 + tr[i]) / 14
            plus_dm_smooth[i] = (prev_plus * 13 + plus_dm[i]) / 14
            minus_dm_smooth[i] = (prev_minus * 13 + minus_dm[i]) / 14
    
    # Calculate DI and DX
    plus_di = np.full_like(atr, np.nan)
    minus_di = np.full_like(atr, np.nan)
    dx = np.full_like(atr, np.nan)
    
    for i in range(len(atr)):
        if i >= 14 and not np.isnan(atr[i]) and atr[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / atr[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / atr[i])
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX: smoothed DX
    adx = np.full_like(dx, np.nan)
    for i in range(len(dx)):
        if i < 27:  # 14 + 13 for smoothing
            continue
        if i == 27:
            adx[i] = np.nanmean(dx[14:i+1]) if not np.all(np.isnan(dx[14:i+1])) else np.nan
        else:
            prev_adx = adx[i-1]
            if not np.isnan(prev_adx) and not np.isnan(dx[i]):
                adx[i] = (prev_adx * 13 + dx[i]) / 14
    
    # Low ADX (<25) indicates ranging/choppy market - we avoid these
    # High ADX (>25) indicates trending - we want this
    adx_filter = adx > 25
    
    # === ALIGN ALL INDICATORS TO 4H TIMEFRAME ===
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter.astype(float))
    
    # Volume average (20-period for 4h) for confirmation
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
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(adx_filter_aligned[i]) or 
            vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2x average (tighter than before)
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Only trade when ADX indicates trending market (avoid chop)
        in_trend = adx_filter_aligned[i] > 0.5
        
        # Entry conditions with 1d EMA50 trend filter
        long_setup = (close[i] > h3_aligned[i]) and vol_confirm and in_trend and (close[i] > ema_50_aligned[i])
        short_setup = (close[i] < l3_aligned[i]) and vol_confirm and in_trend and (close[i] < ema_50_aligned[i])
        
        # Exit conditions: mean reversion to opposite H3/L3 level
        exit_long = close[i] < l3_aligned[i]
        exit_short = close[i] > h3_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals