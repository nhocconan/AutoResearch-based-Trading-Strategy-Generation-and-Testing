#!/usr/bin/env python3
"""
6h_1d_1w_camarilla_range_breakout_v1
Strategy: 6h Camarilla range breakout with 1d/1w trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses daily and weekly Camarilla pivot levels (H4/L4 for breakout, H3/L3 for reversal) with volume confirmation. Trades breakouts only when aligned with higher timeframe trend (1d EMA50 for direction, 1w ADX>25 for trend strength). Avoids false breakouts in ranging markets. Designed to work in both bull (breakouts with trend) and bear (breakouts against weak trend filtered) markets by requiring trend strength confirmation. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_range_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    h4 = c + range_val * 1.1 / 2
    l4 = c - range_val * 1.1 / 2
    h3 = c + range_val * 1.1 / 4
    l3 = c - range_val * 1.1 / 4
    h2 = c + range_val * 1.1 / 6
    l2 = c - range_val * 1.1 / 6
    h1 = c + range_val * 1.1 / 12
    l1 = c - range_val * 1.1 / 12
    return h1, h2, h3, h4, l1, l2, l3, l4

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > high[i-1] - low[i] else 0
        minus_dm[i] = max(high[i-1] - low[i], 0) if high[i-1] - low[i] > high[i] - high[i-1] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    dx = np.zeros_like(high)
    
    atr[period] = np.nansum(tr[1:period+1]) / period
    plus_dm_sum = np.nansum(plus_dm[1:period+1])
    minus_dm_sum = np.nansum(minus_dm[1:period+1])
    
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
    
    adx = np.zeros_like(dx)
    adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
    for i in range(2*period+1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 6h ATR(10) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d Camarilla levels (from previous day)
    h1_1d, h2_1d, h3_1d, h4_1d, l1_1d, l2_1d, l3_1d, l4_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    camarilla_1d = {
        'h1': h1_1d, 'h2': h2_1d, 'h3': h3_1d, 'h4': h4_1d,
        'l1': l1_1d, 'l2': l2_1d, 'l3': l3_1d, 'l4': l4_1d
    }
    
    # 1w Camarilla levels (from previous week)
    h1_1w, h2_1w, h3_1w, h4_1w, l1_1w, l2_1w, l3_1w, l4_1w = calculate_camarilla(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    camarilla_1w = {
        'h1': h1_1w, 'h2': h2_1w, 'h3': h3_1w, 'h4': h4_1w,
        'l1': l1_1w, 'l2': l2_1w, 'l3': l3_1w, 'l4': l4_1w
    }
    
    # 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w ADX for trend strength
    adx_1w = calculate_adx(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values
    )
    
    # Align HTF data to 6h timeframe
    camarilla_1d_aligned = {}
    for key in camarilla_1d:
        camarilla_1d_aligned[key] = align_htf_to_ltf(prices, df_1d, camarilla_1d[key])
    
    camarilla_1w_aligned = {}
    for key in camarilla_1w:
        camarilla_1w_aligned[key] = align_htf_to_ltf(prices, df_1w, camarilla_1w[key])
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_10[i]) or np.isnan(vol_avg[i]) or
            any(np.isnan(v) for v in [
                camarilla_1d_aligned['h3'][i], camarilla_1d_aligned['l3'][i],
                camarilla_1d_aligned['h4'][i], camarilla_1d_aligned['l4'][i],
                camarilla_1w_aligned['h3'][i], camarilla_1w_aligned['l3'][i],
                ema_50_1d_aligned[i], adx_1w_aligned[i]
            ])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        
        # Trend conditions
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        strong_trend = adx_1w_aligned[i] > 25
        
        # Camarilla levels (1d)
        h3_1d = camarilla_1d_aligned['h3'][i]
        l3_1d = camarilla_1d_aligned['l3'][i]
        h4_1d = camarilla_1d_aligned['h4'][i]
        l4_1d = camarilla_1d_aligned['l4'][i]
        
        # Camarilla levels (1w) - for stronger support/resistance
        h3_1w = camarilla_1w_aligned['h3'][i]
        l3_1w = camarilla_1w_aligned['l3'][i]
        
        # Breakout conditions with volume confirmation
        breakout_up = (price_close > h4_1d or price_close > h3_1w) and vol_ratio > 1.5
        breakout_down = (price_close < l4_1d or price_close < l3_1w) and vol_ratio > 1.5
        
        # Reversal conditions at H3/L3 (fade extreme levels)
        reversal_up = (price_close < h3_1d and price_close > l3_1d) and vol_ratio > 1.2
        reversal_down = (price_close > l3_1d and price_close < h3_1d) and vol_ratio > 1.2
        
        # Trading logic
        if breakout_up and uptrend_1d and strong_trend and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and downtrend_1d and strong_trend and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (price_close < camarilla_1d_aligned['h3'][i] or not uptrend_1d):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price_close > camarilla_1d_aligned['l3'][i] or not downtrend_1d):
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals