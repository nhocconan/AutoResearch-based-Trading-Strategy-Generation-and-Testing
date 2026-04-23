#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R(14) with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 AND price > 1d EMA34 AND volume > 1.8x 20-period average.
Short when Williams %R crosses below -20 AND price < 1d EMA34 AND volume > 1.8x 20-period average.
Exit when Williams %R returns to -50 (mean reversion) or ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) and strict volume filter to target 20-50 trades/year.
4h timeframe balances responsiveness with noise reduction for mean reversion in bear markets.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams %R(14) on 4h timeframe
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Williams %R signals: -80 oversold, -20 overbought, -50 midpoint
    williams_r_oversold = -80.0
    williams_r_overbought = -20.0
    williams_r_mid = -50.0
    
    # Volume average (20-period for 4h timeframe)
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
    start_idx = max(34, 14, 20)  # EMA34 needs 34, Williams %R needs 14, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND uptrend (price > EMA34) AND volume spike (1.8x avg)
            if wr_prev <= williams_r_oversold and wr > williams_r_oversold and price > ema34_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Williams %R crosses below -20 (overbought) AND downtrend (price < EMA34) AND volume spike (1.8x avg)
            elif wr_prev >= williams_r_overbought and wr < williams_r_overbought and price < ema34_val and volume[i] > 1.8 * vol_ma_val:
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
            if position == 1 and wr >= williams_r_mid:
                exit_signal = True
            elif position == -1 and wr <= williams_r_mid:
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

name = "4H_WilliamsR_14_1dEMA34_Trend_VolumeConfirmation_MeanReversionExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0