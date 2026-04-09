#!/usr/bin/env python3
# 12h_hma_trend_volume_v1
# Hypothesis: 12h strategy using Hull Moving Average (HMA) trend filter with volume confirmation (>1.5x 20-period average) and 1d HTF trend alignment (price > 20-period EMA). Enters long when price is above HMA(21) with volume confirmation and bullish 1d trend; short when price is below HMA(21) with volume confirmation and bearish 1d trend. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 12-37 trades/year) to work in both bull and bear markets by following volume-confirmed trends aligned with higher timeframe direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = pd.Series(series).ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    wma2 = pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    return hma.values

name = "12h_hma_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # HMA(21) on primary timeframe
    hma_21 = calculate_hma(close, 21)
    
    # 1d HTF trend filter: 20-period EMA on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(hma_21[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below HMA
            if close[i] < hma_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above HMA
            if close[i] > hma_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and 1d trend alignment
            if volume_confirmed:
                # Bullish 1d trend: price above 20-period EMA
                bullish_trend = close[i] > ema_20_1d_aligned[i]
                # Bearish 1d trend: price below 20-period EMA
                bearish_trend = close[i] < ema_20_1d_aligned[i]
                
                # Long: price above HMA with volume and bullish 1d trend
                if close[i] > hma_21[i] and bullish_trend:
                    position = 1
                    signals[i] = 0.25
                # Short: price below HMA with volume and bearish 1d trend
                elif close[i] < hma_21[i] and bearish_trend:
                    position = -1
                    signals[i] = -0.25
    
    return signals