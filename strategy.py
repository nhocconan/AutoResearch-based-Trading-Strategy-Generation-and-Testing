#!/usr/bin/env python3
"""
4h_TRIX_12hEMA34_VolumeSpike_ATRStop
Hypothesis: TRIX (triple exponential moving average) crossing above/below zero signals momentum shifts, filtered by 12h EMA34 trend direction and confirmed by volume spikes. ATR-based stop loss manages risk. Designed for low-to-moderate trade frequency (20-40/year) to avoid fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period standard) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 - np.roll(ema3, 1)
    trix[0] = 0  # first value has no previous
    
    # Get 12h data for EMA34 trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close']
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stop loss (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trix_now = trix[i]
        ema_trend = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        atr_now = atr[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with 12h uptrend and volume spike
            if trix_now > 0 and trix[i-1] <= 0 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TRIX crosses below zero with 12h downtrend and volume spike
            elif trix_now < 0 and trix[i-1] >= 0 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR price hits ATR stop
            if trix_now < 0 or price < entry_price - 2.0 * atr_now:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR price hits ATR stop
            if trix_now > 0 or price > entry_price + 2.0 * atr_now:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0