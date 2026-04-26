#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_HTFVolatilityFilter
Hypothesis: On 4h timeframe, use Camarilla R3/S3 levels from 1d for breakout entries, filtered by 1d trend direction (close > EMA34) and 1w volatility regime (ATR ratio < 0.8 to avoid high volatility chop). Enter long when price breaks above R3 with 1d uptrend and low volatility regime. Enter short when price breaks below S3 with 1d downtrend and low volatility regime. Uses discrete position size 0.25 to balance capture and drawdown. Designed for 19-30 trades/year on 4h by requiring 1d alignment and volatility filter, reducing overtrading while capturing structured moves in both bull and bear markets.
"""

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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for volatility filter (ATR ratio)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r3 = prev_1d_close + 1.1 * camarilla_range / 4
    s3 = prev_1d_close - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate current 4h ATR for volatility ratio
    tr1_4h = np.abs(high - low)
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = np.nan
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility regime: ATR ratio (4h/1w) < 0.8 indicates low volatility regime
    atr_ratio = atr_4h / np.maximum(atr_1w, 1e-10)
    low_vol_regime = atr_ratio < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA warmup, ATR warmup
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(atr_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend alignment
        trend_1d_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_1d_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + 1d uptrend + low volatility regime
            long_signal = (close[i] > r3_aligned[i]) and trend_1d_uptrend and low_vol_regime[i]
            
            # Short: price breaks below S3 + 1d downtrend + low volatility regime
            short_signal = (close[i] < s3_aligned[i]) and trend_1d_downtrend and low_vol_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR 1d trend turns down OR volatility spikes
            if (close[i] < s3_aligned[i] or not trend_1d_uptrend or not low_vol_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR 1d trend turns up OR volatility spikes
            if (close[i] > r3_aligned[i] or not trend_1d_downtrend or not low_vol_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_HTFVolatilityFilter"
timeframe = "4h"
leverage = 1.0