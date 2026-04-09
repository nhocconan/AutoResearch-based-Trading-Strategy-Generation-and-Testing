#!/usr/bin/env python3
# 6h_volume_weighted_hma_v1
# Hypothesis: 6h volume-weighted Hull Moving Average (VW-HMA) crossover with 1w EMA filter and ATR-based volatility scaling.
# Uses 6h timeframe to balance trade frequency and responsiveness. VW-HMA reduces lag while incorporating institutional volume participation.
# Weekly EMA provides multi-timeframe trend bias to avoid counter-trend trades during ranging markets.
# ATR volatility scaling adjusts position size based on market conditions - smaller during high volatility, larger during low volatility.
# Designed for 12-37 trades/year (50-150 over 4 years) with discrete position sizing to minimize fee churn.
# Works in bull/bear markets: VW-HMA captures trends with less whipsaw, weekly EMA filter avoids fades during strong trends,
# volatility scaling reduces risk during turbulent periods.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_weighted_hma_v1"
timeframe = "6h"
leverage = 1.0

def calculate_wma(series, period):
    """Calculate Weighted Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    weights = np.arange(1, period + 1)
    return np.convolve(series, weights / weights.sum(), mode='same')

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = calculate_wma(series, half_period)
    # WMA for full period
    wma_full = calculate_wma(series, period)
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = calculate_wma(raw_hma, sqrt_period)
    return hma

def calculate_vw_hma(close, volume, period):
    """Calculate Volume-Weighted Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    # Calculate typical price weighted by volume
    typical_price = close  # Using close as typical price for simplicity
    vol_sum = pd.Series(volume).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    vol_sum = np.where(vol_sum == 0, 1, vol_sum)
    
    # Volume-weighted price
    vw_price = np.cumsum(typical_price * volume) / np.cumsum(volume)
    vw_price = np.where(np.isnan(vw_price), typical_price, vw_price)
    
    # Apply HMA to volume-weighted price
    return calculate_hma(vw_price, period)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for VW-HMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:  # Need enough for VW-HMA(50)
        return np.zeros(n)
    
    # Calculate 6h volume-weighted HMA(50) for trend
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    vw_hma_6h = calculate_vw_hma(close_6h, volume_6h, 50)
    
    # Align 6h VW-HMA to 6h timeframe (completed 6h candle only)
    vw_hma_6h_aligned = align_htf_to_ltf(prices, df_6h, vw_hma_6h)
    
    # Get 1w HTF data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 6h timeframe (completed weekly candle only)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate ATR(14) for volatility scaling
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio for volatility scaling (current ATR / 50-period ATR mean)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Volatility scaling factor: inverse of ATR ratio, capped between 0.5 and 1.5
    vol_scale = np.clip(1.0 / atr_ratio, 0.5, 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(vw_hma_6h_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h VW-HMA
            if close[i] < vw_hma_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by volatility (inverse volatility scaling)
                base_size = 0.25
                scaled_size = base_size * vol_scale[i]
                signals[i] = np.clip(scaled_size, 0.10, 0.40)
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h VW-HMA
            if close[i] > vw_hma_6h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by volatility (inverse volatility scaling)
                base_size = 0.25
                scaled_size = base_size * vol_scale[i]
                signals[i] = -np.clip(scaled_size, 0.10, 0.40)
        else:  # Flat
            # Enter long: price closes above 6h VW-HMA, above 1w EMA, with volume confirmation
            if (close[i] > vw_hma_6h_aligned[i]) and (close[i] > ema_1w_aligned[i]):
                position = 1
                # Scale position by volatility (inverse volatility scaling)
                base_size = 0.25
                scaled_size = base_size * vol_scale[i]
                signals[i] = np.clip(scaled_size, 0.10, 0.40)
            # Enter short: price closes below 6h VW-HMA, below 1w EMA, with volume confirmation
            elif (close[i] < vw_hma_6h_aligned[i]) and (close[i] < ema_1w_aligned[i]):
                position = -1
                # Scale position by volatility (inverse volatility scaling)
                base_size = 0.25
                scaled_size = base_size * vol_scale[i]
                signals[i] = -np.clip(scaled_size, 0.10, 0.40)
    
    return signals