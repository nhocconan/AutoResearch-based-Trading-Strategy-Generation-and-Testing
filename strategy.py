#!/usr/bin/env python3
# 6h_ADX_Trend_With_Volume_Pullback
# Hypothesis: ADX identifies strong trends (ADX>25), while pullbacks to VWAP during these trends offer high-probability entries.
# Uses 1d trend filter (EMA50) to align with higher timeframe direction, reducing counter-trend trades.
# Volume spike confirms breakout strength. Designed for 6h timeframe with moderate frequency (target: 15-35 trades/year).
# Works in bull markets via trend continuation and in bear markets via short-side pullbacks.

name = "6h_ADX_Trend_With_Volume_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 6h data
    # True Range (TR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first period
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Calculate VWAP (reset daily)
    # Typical price
    tp = (high + low + close) / 3.0
    # Cumulative values
    cum_vol = np.cumsum(volume)
    cum_tpv = np.cumsum(tp * volume)
    vwap = cum_tpv / cum_vol
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX (14+14=28), VWAP (need volume), EMA50 (50)
    start_idx = max(28, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx[i]) or np.isnan(vwap[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        # Trend direction from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to VWAP (pullback to mean)
        near_vwap = np.abs(close[i] - vwap[i]) < (0.01 * vwap[i])  # within 1% of VWAP
        
        if position == 0:
            # Long entry: strong uptrend + pullback to VWAP + volume spike
            if strong_trend and uptrend and near_vwap and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: strong downtrend + pullback to VWAP + volume spike
            elif strong_trend and downtrend and near_vwap and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakens or price moves too far from VWAP
            if not strong_trend or not uptrend or np.abs(close[i] - vwap[i]) > (0.02 * vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens or price moves too far from VWAP
            if not strong_trend or not downtrend or np.abs(close[i] - vwap[i]) > (0.02 * vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals