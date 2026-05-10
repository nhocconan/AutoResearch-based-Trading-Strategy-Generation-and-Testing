#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Trade Camarilla pivot reversals on 12h with daily trend filter and volume confirmation.
# Long when: price breaks above Camarilla R1 with daily uptrend and volume > 1.5x average.
# Short when: price breaks below Camarilla S1 with daily downtrend and volume > 1.5x average.
# Uses 1w regime filter: only trade when weekly ADX < 25 (range market) to avoid whipsaws.
# Works in bull/bear by fading extremes in ranging markets while respecting daily trend.
# Target: 12-37 trades/year per symbol.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Weekly regime filter (ADX < 25 = ranging market)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
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
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    ranging_market = adx_1w < 25
    
    # Align weekly ranging to 12h
    ranging_market_aligned = align_htf_to_ltf(prices, df_1w, ranging_market.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(ranging_market_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        # Need previous day's OHLC - we'll approximate using 12h bars
        # For simplicity, use 24-period lookback (2 days of 12h data)
        if i >= 24:
            lookback_high = np.max(high[i-24:i])
            lookback_low = np.min(low[i-24:i])
            lookback_close = close[i-1]  # previous close
            
            # Camarilla equations
            range_val = lookback_high - lookback_low
            r1 = lookback_close + (range_val * 1.1 / 12)
            s1 = lookback_close - (range_val * 1.1 / 12)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        ranging = ranging_market_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + ranging market + price breaks above R1 + volume
            if daily_up and ranging and close[i] > r1 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + ranging market + price breaks below S1 + volume
            elif daily_down and ranging and close[i] < s1 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price moves below R1 or trend changes
            if close[i] < r1 or not daily_up or not ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price moves above S1 or trend changes
            if close[i] > s1 or not daily_down or not ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals