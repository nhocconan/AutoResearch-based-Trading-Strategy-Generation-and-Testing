#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Breakout with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 AND volume > 1.5x 20-period average volume.
Short when Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 AND volume > 1.5x 20-period average volume.
Exit when Williams %R returns to -50 (mean reversion) OR ATR trailing stop (2.0*ATR from extreme).
Williams %R identifies overextended conditions; EMA34 filters for trend alignment; volume confirms momentum.
Works in both bull (bounce from oversold) and bear (rejection from overbought) markets by capturing mean reversion within trend.
Target: ~20-30 trades/year on 4h timeframe with discrete sizing 0.25.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34)  # williams_r14, vol_ma20, ema_34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        # Williams %R crossovers (using previous bar to avoid look-ahead)
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND uptrend (price > EMA34) AND volume spike
            if wr > -80 and wr_prev <= -80 and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Williams %R crosses below -20 (overbought rejection) AND downtrend (price < EMA34) AND volume spike
            elif wr < -20 and wr_prev >= -20 and price < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R returns to -50 (mean reversion)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dEMA34_Trend_VolumeSpike_MeanReversion_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0