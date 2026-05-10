#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hTrend_Volume_Regime
# Hypothesis: Buy at Camarilla R1 in 12h uptrend with volume spike in choppy markets (mean reversion).
# Short at Camarilla S1 in 12h downtrend with volume spike in choppy markets.
# Uses 4h ADX<30 for chop regime detection. Works in bull/bear by fading extremes in ranges.
# Target: 25-40 trades/year per symbol.

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_Volume_Regime"
timeframe = "4h"
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
    
    # 4h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Volatility for stop (ATR 14)
    high_low = high[1:] - low[1:]
    high_close = np.abs(high[1:] - close[:-1])
    low_close = np.abs(low[1:] - close[:-1])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr = np.concatenate([[0], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX for chop regime (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = np.concatenate([np.zeros(1), plus_di])
    minus_di = np.concatenate([np.zeros(1), minus_di])
    dx = np.concatenate([np.zeros(1), dx])
    adx = np.concatenate([np.zeros(1), adx])
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Camarilla levels from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        chop_regime = adx[i] < 30  # choppy market
        
        trend_up = trend_12h_up_aligned[i] > 0.5
        trend_down = trend_12h_down_aligned[i] > 0.5
        
        if position == 0:
            # Long at R1 in 12h uptrend with volume spike in chop
            if trend_up and volume_spike and chop_regime:
                if close[i] >= R1_aligned[i] * 0.999 and close[i] <= R1_aligned[i] * 1.001:
                    signals[i] = 0.25
                    position = 1
            # Short at S1 in 12h downtrend with volume spike in chop
            elif trend_down and volume_spike and chop_regime:
                if close[i] <= S1_aligned[i] * 1.001 and close[i] >= S1_aligned[i] * 0.999:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: trend breaks or volatility expands
            if not trend_up or adx[i] > 35 or close[i] < R1_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend breaks or volatility expands
            if not trend_down or adx[i] > 35 or close[i] > S1_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals