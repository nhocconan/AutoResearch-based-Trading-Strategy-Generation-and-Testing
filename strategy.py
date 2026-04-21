#!/usr/bin/env python3
"""
1h_HTFTrend_VolumeSpike_Entry
Hypothesis: Use 4h EMA50 for trend direction and 1d ATR ratio for volatility regime filter.
Enter on 1h bullish/bearish engulfing candles with volume spike (>1.5x 20-bar average).
Only trade during 08-20 UTC session to avoid low-liquidity hours.
Fixed size 0.20 to limit drawdown. Target 15-35 trades/year.
Works in bull/bear: trend filter avoids counter-trend trades, volume spike ensures conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ATR ratio for volatility regime (avoid choppy markets)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_30_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_10_1d / (atr_30_1d + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h indicators for entry timing
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    open_1h = prices['open'].values
    volume_1h = prices['volume'].values
    
    # Volume spike: >1.5x 20-bar average
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1h > (1.5 * vol_ma_20)
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bull_engulf = (close_1h > open_1h) & (open_1h < close_1h) & \
                  (close_1h >= open_1h) & (open_1h <= close_1h) & \
                  (close_1h > open_1h.shift(1)) & (open_1h < close_1h.shift(1)) & \
                  (close_1h >= open_1h.shift(1)) & (open_1h <= close_1h.shift(1)) & \
                  (close_1h - open_1h) >= (close_1h.shift(1) - open_1h.shift(1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bear_engulf = (close_1h < open_1h) & (open_1h > close_1h) & \
                  (close_1h <= open_1h) & (open_1h >= close_1h) & \
                  (close_1h < open_1h.shift(1)) & (open_1h > close_1h.shift(1)) & \
                  (close_1h <= open_1h.shift(1)) & (open_1h >= close_1h.shift(1)) & \
                  (open_1h - close_1h) >= (open_1h.shift(1) - close_1h.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) 
            or np.isnan(volume_spike[i]) or np.isnan(bull_engulf[i]) 
            or np.isnan(bear_engulf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: avoid extreme volatility (ATR ratio > 1.5) or low volatility (< 0.8)
        if atr_ratio_aligned[i] > 1.5 or atr_ratio_aligned[i] < 0.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1h[i]
        
        if position == 0:
            # Long: Uptrend (price > 4h EMA50) + bullish engulfing + volume spike
            if price > ema_50_4h_aligned[i] and bull_engulf[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend (price < 4h EMA50) + bearish engulfing + volume spike
            elif price < ema_50_4h_aligned[i] and bear_engulf[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price crosses below 4h EMA50 or opposite engulfing candle
            if price < ema_50_4h_aligned[i] or bear_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price crosses above 4h EMA50 or opposite engulfing candle
            if price > ema_50_4h_aligned[i] or bull_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HTFTrend_VolumeSpike_Entry"
timeframe = "1h"
leverage = 1.0