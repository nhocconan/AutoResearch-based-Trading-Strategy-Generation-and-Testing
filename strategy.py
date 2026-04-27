#!/usr/bin/env python3
"""
1h Breakout Strategy with 4h/1d Regime Filter for BTC/ETH
Hypothesis: Use 4h/1d price action to filter regime (trending vs ranging) and direction,
while using 1h for precise breakout entry timing. In trending markets (ADX>25),
trade breakouts in direction of trend; in ranging markets (ADX<20), fade extremes.
This reduces false breakouts and works in both bull/bear markets by adapting to regime.
Target: 15-37 trades/year (60-150 over 4 years) with disciplined entry.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from mtf_data import get_htf_data, align_htf_to_ltf

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index - measures trend strength"""
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = np.zeros(n)
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # DI+ and DI-
    plus_di = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_vals = np.full(n, np.nan)
    adx_vals[2*period-2] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period-1, n):
        adx_vals[i] = (adx_vals[i-1] * (period-1) + dx[i]) / period
    
    return adx_vals

def donchian_channels(high: np.ndarray, low: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Donchian Channels - upper and lower bands"""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(low), np.nan)
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1h data for regime and direction filtering
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    # 4h ADX for trend strength (regime filter)
    adx_4h = adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d EMA50 for long-term trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h Donchian channels for breakout signals
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    # Volume filter - avoid low-volume breakouts
    vol_ma_20 = np.zeros(n)
    vol_ma_20[:] = np.nan
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Session filter: 08-20 UTC (reduce noise)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup period
    start_idx = max(35, 20, 19)  # ADX needs 2*period, Donchian needs period, volume needs 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        adx_now = adx_4h_aligned[i]
        ema_50_now = ema_50_1d_aligned[i]
        
        # Regime determination
        is_trending = adx_now > 25
        is_ranging = adx_now < 20
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Determine trade direction based on regime
            if is_trending:
                # In trending markets, trade with trend direction
                if price_now > ema_50_now:
                    trend_bias = 1  # Long bias
                else:
                    trend_bias = -1  # Short bias
                
                # Enter on breakout in trend direction
                if trend_bias == 1 and price_now > donch_hi[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif trend_bias == -1 and price_now < donch_lo[i] and vol_filter:
                    signals[i] = -size
                    position = -1
            elif is_ranging:
                # In ranging markets, fade extremes (mean reversion)
                # Simple approach: buy near lower band, sell near upper band
                if price_now <= donch_lo[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif price_now >= donch_hi[i] and vol_filter:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle or reverse signal
            donch_mid = (donch_hi[i] + donch_lo[i]) / 2
            if price_now < donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            donch_mid = (donch_hi[i] + donch_lo[i]) / 2
            if price_now > donch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Breakout_Regime_ADX_EMA50"
timeframe = "1h"
leverage = 1.0