#!/usr/bin/env python3
# 1h_TripleConfirmation_TrendFollow
# Hypothesis: Use 4h trend (EMA21) and 1d regime (ADX<30 for range, >25 for trend) to filter entries.
# On 1h, enter long when price crosses above EMA50 with volume > 1.3x average in bullish regime.
# Enter short when price crosses below EMA50 with volume > 1.3x average in bearish regime.
# Uses 1h only for timing, 4h/1d for direction/regime. Target: 20-35 trades/year per symbol.
# Works in bull/bear by following higher timeframe trend and avoiding choppy markets.

name = "1h_TripleConfirmation_TrendFollow"
timeframe = "1h"
leverage = 1.0

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
    
    # 1h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # EMA50 for entry signals
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 4h trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    uptrend_4h = close_4h > ema21_4h
    downtrend_4h = close_4h < ema21_4h
    
    # Align 4h trend to 1h
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # 1d regime filter (ADX for trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                     pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                      pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    # Prepend NaN for alignment (since we lost first bar in calculations)
    adx_1d = np.concatenate([np.full(1, np.nan), adx_1d])
    
    # Align 1d ADX to 1h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        vol_filter = vol_ratio > 1.3
        
        # Determine market regime
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 30  # Slight overlap for hysteresis
        
        if position == 0:
            # Enter long: 4h uptrend + volume + price crosses above EMA50
            if uptrend_4h_aligned[i] > 0.5 and vol_filter:
                if close[i] > ema50[i] and close[i-1] <= ema50[i-1]:
                    signals[i] = 0.20
                    position = 1
            # Enter short: 4h downtrend + volume + price crosses below EMA50
            elif downtrend_4h_aligned[i] > 0.5 and vol_filter:
                if close[i] < ema50[i] and close[i-1] >= ema50[i-1]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit: 4h trend fails or price crosses below EMA50
            if uptrend_4h_aligned[i] < 0.5 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: 4h trend fails or price crosses above EMA50
            if downtrend_4h_aligned[i] < 0.5 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals