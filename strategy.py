#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Mean Reversion with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA50 (uptrend) AND 6h volume > 1.5x 20-period average volume.
Short when Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA50 (downtrend) AND 6h volume > 1.5x 20-period average volume.
Exit when Williams %R crosses -50 (mean reversion) OR ATR trailing stop (2.5*ATR from extreme).
Williams %R identifies overextended moves; EMA50 filters for trend alignment; volume confirms reversal strength.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
Target: ~20-30 trades/year on 6h timeframe with discrete sizing 0.25.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6h Williams %R(14)
    lr = 14
    highest_high = pd.Series(high).rolling(window=lr, min_periods=lr).max().values
    lowest_low = pd.Series(low).rolling(window=lr, min_periods=lr).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 6h volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
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
    start_idx = max(lr, 20, 50)  # wr14, vol_ma20, ema_50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr_val = wr[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: WR crosses above -80 (oversold) AND uptrend (price > EMA50) AND volume spike
            if wr_val > -80 and wr[i-1] <= -80 and price > ema_50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: WR crosses below -20 (overbought) AND downtrend (price < EMA50) AND volume spike
            elif wr_val < -20 and wr[i-1] >= -20 and price < ema_50_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: WR crosses -50 (mean reversion)
            if position == 1 and wr_val < -50 and wr[i-1] >= -50:
                exit_signal = True
            elif position == -1 and wr_val > -50 and wr[i-1] <= -50:
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

name = "6H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirmation_EXIT_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0