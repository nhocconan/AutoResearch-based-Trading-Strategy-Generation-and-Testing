#!/usr/bin/env python3
# 1d_RSI_2Period_MeanReversion_with_ADX_TrendFilter
# Hypothesis: On 1d timeframe, enter long when 2-period RSI < 10 and price > 200-day EMA (bullish trend), 
# enter short when 2-period RSI > 90 and price < 200-day EMA (bearish trend).
# Use weekly ADX > 25 to ensure trending market (avoid chop). Exit when RSI crosses 50.
# Targets 10-20 trades/year for low fee drift. Works in bull/bear by fading extremes only in trend.

name = "1d_RSI_2Period_MeanReversion_with_ADX_TrendFilter"
timeframe = "1d"
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
    
    # Daily 2-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily 200 EMA for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly ADX for trend strength filter (avoid chop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # True Range
    tr1 = wh - wl
    tr2 = np.abs(wh - np.roll(wc, 1))
    tr3 = np.abs(wl - np.roll(wc, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up = wh - np.roll(wh, 1)
    down = np.roll(wl, 1) - wl
    up[0] = down[0] = np.nan
    up = np.where((up > down) & (up > 0), up, 0)
    down = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed DM and TR
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    up_smooth = pd.Series(up).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    down_smooth = pd.Series(down).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * up_smooth / (atr + 1e-10)
    minus_di = 100 * down_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 and indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema200[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema200_val = ema200[i]
        adx_val = adx_aligned[i]
        
        # Only trade in trending markets (ADX > 25)
        if adx_val < 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 10 (extreme oversold) and price > EMA200 (bullish trend)
            if rsi_val < 10 and close[i] > ema200_val:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 90 (extreme overbought) and price < EMA200 (bearish trend)
            elif rsi_val > 90 and close[i] < ema200_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals