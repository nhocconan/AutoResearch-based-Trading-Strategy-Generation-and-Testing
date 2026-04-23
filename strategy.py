#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Mean Reversion with 12h EMA50 trend filter and volume confirmation.
Long when Williams %R(14) < -80 (oversold) AND close > 12h EMA50 (uptrend) AND volume > 1.5x 20-period average.
Short when Williams %R(14) > -20 (overbought) AND close < 12h EMA50 (downtrend) AND volume > 1.5x 20-period average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ATR trailing stop (2.0*ATR from extreme).
Williams %R is effective in ranging markets which dominate BTC/ETH in 2025+ test period.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams %R(14) calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    dd = highest_high - lowest_low
    dd = np.where(dd == 0, 1e-10, dd)
    willr = -100 * (highest_high - close) / dd
    
    # Volume average (20-period)
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
    start_idx = max(50, 14, 20, 14)  # EMA50 needs 50, Williams %R needs 14, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(willr[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_12h_aligned[i]
        willr_val = willr[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend (price > EMA50) AND volume spike (1.5x avg)
            if willr_val < -80 and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Williams %R overbought (> -20) AND downtrend (price < EMA50) AND volume spike (1.5x avg)
            elif willr_val > -20 and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            if position == 1 and willr_val > -50:
                exit_signal = True
            elif position == -1 and willr_val < -50:
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

name = "4H_WilliamsR_MeanReversion_12hEMA50_Trend_VolumeConfirmation_WR50Exit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0