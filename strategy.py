#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R Mean Reversion with 1w EMA50 trend filter and volume confirmation.
Long when Williams %R(14) < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-day average volume.
Short when Williams %R(14) > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-day average volume.
Exit when Williams %R returns to -50 (mean reversion) OR ATR trailing stop (2.5*ATR from extreme).
Williams %R identifies overextended moves; 1w EMA50 filters for higher-timeframe trend alignment; volume confirms conviction.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) by fading extremes with trend.
Target: ~10-20 trades/year on 1d timeframe with discrete sizing 0.25.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R(14) on 1d
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # 20-day volume average for spike filter
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
    start_idx = max(14, 20, 50)  # williams_r14, vol_ma20, ema_50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Oversold AND uptrend (price > 1w EMA50) AND volume spike
            if wr < -80 and price > ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Overbought AND downtrend (price < 1w EMA50) AND volume spike
            elif wr > -20 and price < ema_trend and volume[i] > 1.5 * vol_ma_val:
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
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WilliamsR_MeanReversion_1wEMA50_Trend_VolumeSpike_MeanReversion_ATRTrailingStop"
timeframe = "1d"
leverage = 1.0