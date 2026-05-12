#!/usr/bin/env python3
# 6h_ADX_VWAP_Pullback_1dTrend
# Hypothesis: Pullbacks to VWAP during strong trends (ADX>25) on 6h, filtered by 1d EMA trend.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Uses ADX for trend strength and VWAP for mean reversion within trend.
# Target: 20-40 trades/year.

name = "6h_ADX_VWAP_Pullback_1dTrend"
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
    volume = prices['volume'].values
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ADX(14) for trend strength ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / atr_14
    di_minus = 100 * dm_minus_14 / atr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === VWAP (typical price * volume) / cumulative volume ===
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = cum_pv / cum_vol
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Strong trend filter
        strong_trend = adx[i] > 25
        
        # Distance from VWAP (normalized by price)
        dist_from_vwap = (close[i] - vwap[i]) / vwap[i]
        
        if position == 0:
            # LONG: Pullback to VWAP in strong uptrend
            if (trend_up and strong_trend and dist_from_vwap < -0.005):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to VWAP in strong downtrend
            elif (trend_down and strong_trend and dist_from_vwap > 0.005):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend weakens or price goes above VWAP
            if (not trend_up or not strong_trend or dist_from_vwap > 0.002):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens or price goes below VWAP
            if (not trend_down or not strong_trend or dist_from_vwap < -0.002):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals