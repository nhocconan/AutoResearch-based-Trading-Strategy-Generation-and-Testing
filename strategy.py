#!/usr/bin/env python3
"""
4h Triangular Moving Average (TMA) Pullback with Volume Confirmation and Weekly Trend Filter
Hypothesis: TMA acts as dynamic support/resistance. Pullbacks to TMA with volume confirmation
and weekly trend alignment capture high-probability mean-reversion bounces in both bull and bear markets.
Designed for 15-35 trades/year on 4h timeframe. Uses weekly EMA50 for trend filter to avoid counter-trend trades.
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
    
    # Get weekly data for EMA trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend filter
    ema_50_w = pd.Series(df_w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_w, ema_50_w)
    
    # Triangular Moving Average (TMA) on 4h: SMA of SMA
    # TMA(20) = SMA(SMA(close, 10), 10)
    sma1 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    tma = pd.Series(sma1).rolling(window=10, min_periods=10).mean().values
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (4h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # enough for TMA calculation
    
    for i in range(start_idx, n):
        if (np.isnan(tma[i]) or 
            np.isnan(ema_50_w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tma_val = tma[i]
        ema_50_w = ema_50_w_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: pullback to TMA support in uptrend (price above weekly EMA50)
            if price > ema_50_w and price <= tma_val + 0.1 * atr_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: pullback to TMA resistance in downtrend (price below weekly EMA50)
            elif price < ema_50_w and price >= tma_val - 0.1 * atr_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price moves above TMA resistance or ATR trailing stop
            if price >= tma_val + 0.2 * atr_val or price < (high[i] - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price moves below TMA support or ATR trailing stop
            if price <= tma_val - 0.2 * atr_val or price > (low[i] + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TMA_Pullback_VolumeSpike_WeeklyEMA50"
timeframe = "4h"
leverage = 1.0