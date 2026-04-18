#!/usr/bin/env python3
"""
1h_Flipping_RSI_Band_With_TrendFilter
Hypothesis: In 1h, use RSI(14) with dynamic bands (40/60) for mean reversion in range,
and trend filter from 4h EMA50 to capture directional moves. Works in bull (trend follow)
and bear (mean revert in range) by switching behavior based on 4h trend strength (ADX).
Designed for ~25 trades/year to avoid fee drag.
"""

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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_4h_raw = pd.Series(close_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values  # placeholder, will compute properly
    # Recompute ADX properly on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    plus_dm_4h = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), np.maximum(h_4h[1:] - h_4h[:-1], 0), 0)
    minus_dm_4h = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), np.maximum(l_4h[1:] - l_4h[:-1], 0), 0)
    plus_dm_4h = np.insert(plus_dm_4h, 0, 0)
    minus_dm_4h = np.insert(minus_dm_4h, 0, 0)
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h_arr, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h_arr, 1))
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_4h = 100 * pd.Series(plus_dm_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_4h + 1e-10)
    minus_di_4h = 100 * pd.Series(minus_dm_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_4h + 1e-10)
    dx_4h = (np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h + 1e-10)) * 100
    adx_4h = pd.Series(dx_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 14)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(adx_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema4h = ema_50_4h_aligned[i]
        adx4h = adx_4h_aligned[i]
        price = close[i]
        
        # Regime: ADX > 25 = trending, ADX < 20 = ranging (with hysteresis)
        if position == 0:
            if adx4h > 25:
                # Trending: follow 4h EMA
                if price > ema4h and rsi_val > 50:
                    signals[i] = 0.20
                    position = 1
                elif price < ema4h and rsi_val < 50:
                    signals[i] = -0.20
                    position = -1
            else:
                # Ranging: mean revert at RSI extremes
                if rsi_val < 40:
                    signals[i] = 0.20
                    position = 1
                elif rsi_val > 60:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: trend fails or RSI overbought in range
            if adx4h > 25:
                if price < ema4h or rsi_val > 70:
                    signals[i] = 0.0
                    position = 0
            else:
                if rsi_val > 60:
                    signals[i] = 0.0
                    position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: trend fails or RSI oversold in range
            if adx4h > 25:
                if price > ema4h or rsi_val < 30:
                    signals[i] = 0.0
                    position = 0
            else:
                if rsi_val < 40:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1h_Flipping_RSI_Band_With_TrendFilter"
timeframe = "1h"
leverage = 1.0