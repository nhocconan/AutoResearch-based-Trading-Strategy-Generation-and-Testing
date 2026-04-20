#!/usr/bin/env python3
# 1h_MultiTimeframe_Trend_Pullback
# Hypothesis: In trending markets (4h ADX > 25), buy pullbacks to EMA20 on 1h with volume confirmation.
# In ranging markets (4h ADX < 20), fade extremes at Bollinger Bands (20,2) with RSI divergence.
# Uses 4h for regime and trend direction, 1h for precise entry timing.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.

name = "1h_MultiTimeframe_Trend_Pullback"
timeframe = "1h"
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
    
    # Get 4h data for regime and trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA20 for trend direction
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h ADX20 for regime/trend strength
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: EMA with alpha=1/period
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if np.isnan(result[i-1]):
                    result[i] = data[i]
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = WilderSmooth(tr, 20)
    dm_plus_smooth = WilderSmooth(dm_plus, 20)
    dm_minus_smooth = WilderSmooth(dm_minus, 20)
    
    # Directional Indicators
    di_plus = np.full_like(high_4h, np.nan)
    di_minus = np.full_like(high_4h, np.nan)
    dx = np.full_like(high_4h, np.nan)
    
    valid = tr_smooth != 0
    di_plus[valid] = 100 * dm_plus_smooth[valid] / tr_smooth[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / tr_smooth[valid]
    dx[valid] = 100 * np.abs(di_plus[valid] - di_minus[valid]) / (di_plus[valid] + di_minus[valid])
    
    # ADX: Wilder smoothed DX
    adx_4h = WilderSmooth(dx, 20)
    
    # Align 4h indicators to 1h
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1h indicators for entry timing
    # EMA20 for pullback entries
    ema20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Bollinger Bands (20,2) for mean reversion in ranging markets
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    # RSI(14) for divergence signals
    def RSI(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder smoothing
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(close) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = RSI(close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure indicators are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]) or
            np.isnan(ema20_1h[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(rsi_1h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market (ADX > 25): trade pullbacks in direction of 4h trend
            if adx_4h_aligned[i] > 25:
                # Uptrend: price > 4h EMA20
                if close[i] > ema20_4h_aligned[i]:
                    # Long: pullback to 1h EMA20 with volume
                    if close[i] <= ema20_1h[i] * 1.005 and volume_filter[i]:  # Within 0.5% of EMA20
                        signals[i] = 0.20
                        position = 1
                # Downtrend: price < 4h EMA20
                elif close[i] < ema20_4h_aligned[i]:
                    # Short: pullback to 1h EMA20 with volume
                    if close[i] >= ema20_1h[i] * 0.995 and volume_filter[i]:  # Within 0.5% of EMA20
                        signals[i] = -0.20
                        position = -1
            # Ranging market (ADX < 20): fade extremes at Bollinger Bands
            elif adx_4h_aligned[i] < 20:
                # Long: price at lower BB with RSI < 30 (oversold)
                if close[i] <= bb_lower[i] and rsi_1h[i] < 30 and volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price at upper BB with RSI > 70 (overbought)
                elif close[i] >= bb_upper[i] and rsi_1h[i] > 70 and volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            # Exit if trend reverses (ADX > 25 and price crosses 4h EMA20 opposite direction)
            if adx_4h_aligned[i] > 25 and close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Exit if RSI overbought in ranging market
            elif adx_4h_aligned[i] < 20 and rsi_1h[i] > 70:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks Bollinger Band in opposite direction
            elif adx_4h_aligned[i] < 20 and close[i] >= bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit conditions
            # Exit if trend reverses (ADX > 25 and price crosses 4h EMA20 opposite direction)
            if adx_4h_aligned[i] > 25 and close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Exit if RSI oversold in ranging market
            elif adx_4h_aligned[i] < 20 and rsi_1h[i] < 30:
                signals[i] = 0.0
                position = 0
            # Exit if price breaks Bollinger Band in opposite direction
            elif adx_4h_aligned[i] < 20 and close[i] <= bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals