#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R (14) mean reversion with 1w EMA50 trend filter and volume confirmation.
Long when Williams %R crosses above -80 (oversold) AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when Williams %R crosses below -20 (overbought) AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when Williams %R crosses -50 (mean reversion midpoint) OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~15-25 trades/year on 6h timeframe.
Williams %R is an effective momentum oscillator that identifies overextended moves; mean reversion at extremes with trend alignment and volume confirmation captures high-probability reversals.
Works in bull (buying oversold dips in uptrend) and bear (selling overbought rallies in downtrend) markets.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R (14) on 1d timeframe for more stable signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
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
    start_idx = max(50, 20, 14, 14)  # EMA50 needs 50, vol MA needs 20, ATR needs 14, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_1w_aligned[i]
        wr_val = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) AND uptrend (price > EMA50) AND volume spike (1.5x avg)
            if wr_val > -80 and williams_r_aligned[i-1] <= -80 and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Williams %R crosses below -20 (overbought) AND downtrend (price < EMA50) AND volume spike (1.5x avg)
            elif wr_val < -20 and williams_r_aligned[i-1] >= -20 and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: Williams %R crosses -50 (mean reversion midpoint)
            if position == 1 and wr_val < -50 and williams_r_aligned[i-1] >= -50:
                exit_signal = True
            elif position == -1 and wr_val > -50 and williams_r_aligned[i-1] <= -50:
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

name = "6H_WilliamsR_14_1wEMA50_Trend_VolumeConfirmation_WR50Exit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0