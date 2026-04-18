#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Weekly EMA20 Trend + Volume Spike + ATR Stop
Hypothesis: Price breaking out of 20-day channel with volume and weekly trend alignment captures breakouts in both bull and bear markets.
Uses weekly EMA20 as trend filter to avoid counter-trend trades, volume spike for confirmation, and ATR-based stop to manage risk.
Designed for low trade frequency (target: 10-30 trades/year) with clear edge in trending and ranging markets.
"""

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
    
    # Get weekly data for EMA20 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate ATR(14) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel (20-day high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (2x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above weekly EMA20
            if (price > donch_high[i] and volume_spike[i] and price > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low with volume spike and below weekly EMA20
            elif (price < donch_low[i] and volume_spike[i] and price < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: hold until stop loss or reversal
            signals[i] = 0.25
            # Stop loss: 2 * ATR below entry
            if price <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Optional exit: price closes below weekly EMA20 (trend change)
            elif price < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until stop loss or reversal
            signals[i] = -0.25
            # Stop loss: 2 * ATR above entry
            if price >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Optional exit: price closes above weekly EMA20 (trend change)
            elif price > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_WeeklyEMA20_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0