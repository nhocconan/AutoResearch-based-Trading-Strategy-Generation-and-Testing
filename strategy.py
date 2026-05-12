#!/usr/bin/env python3
# 6h_OrderBlock_1wTrend_Volume
# Hypothesis: Identify institutional order blocks (consolidation zones) on 6h and trade breakouts in the direction of 1w trend with volume confirmation.
# Order blocks form during accumulation/distribution; breakouts often lead to sustained moves.
# Uses 1w trend filter to avoid counter-trend trades, reducing whipsaws in sideways markets.
# Designed for low frequency (15-30 trades/year) by requiring confluence of structure, trend, and volume.

name = "6h_OrderBlock_1wTrend_Volume"
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
    
    # === 6h data for order block detection ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Identify order blocks: look for strong candles followed by consolidation
    # Bullish OB: strong down candle followed by consolidation and then up move
    # Bearish OB: strong up candle followed by consolidation and then down move
    body_6h = np.abs(close_6h - np.roll(open_6h, 1)) if 'open' in df_6h else np.abs(close_6h - np.roll(close_6h, 1))
    # Use high-low range as proxy for body if open not available
    if 'open' not in df_6h:
        body_6h = np.abs(close_6h - np.roll(close_6h, 1))
    else:
        open_6h = df_6h['open'].values
        body_6h = np.abs(close_6h - open_6h)
    
    # Calculate average body size
    avg_body = pd.Series(body_6h).rolling(window=20, min_periods=20).mean().values
    
    # Identify potential order block zones: high volatility candles
    strong_candle = body_6h > 1.5 * avg_body
    
    # Define order block zones as the range of the strong candle
    ob_high = np.where(strong_candle, high_6h, np.nan)
    ob_low = np.where(strong_candle, low_6h, np.nan)
    
    # Forward fill to maintain zone until broken
    ob_high_series = pd.Series(ob_high)
    ob_low_series = pd.Series(ob_low)
    ob_high_ffill = ob_high_series.ffill().values
    ob_low_ffill = ob_low_series.ffill().values
    
    # Align to lower timeframe
    ob_high_aligned = align_htf_to_ltf(prices, df_6h, ob_high_ffill)
    ob_low_aligned = align_htf_to_ltf(prices, df_6h, ob_low_ffill)
    
    # === 1w data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(50) on 1w for long-term trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ob_high_aligned[i]) or np.isnan(ob_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above order block high with volume, in uptrend
            if close[i] > ob_high_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below order block low with volume, in downtrend
            elif close[i] < ob_low_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to order block low or trend reverses
            if close[i] <= ob_low_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to order block high or trend reverses
            if close[i] >= ob_high_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals