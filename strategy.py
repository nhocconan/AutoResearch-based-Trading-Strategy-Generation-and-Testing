#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA50 trend filter + volume confirmation.
Long when Williams %R < -80 (oversold) AND close > 1d EMA50 AND volume > 1.8x 20-period average.
Short when Williams %R > -20 (overbought) AND close < 1d EMA50 AND volume > 1.8x 20-period average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profit potential.
Williams %R captures short-term overextensions that reverse in both bull and bear markets.
1d EMA50 filter ensures alignment with longer-term trend, reducing counter-trend whipsaws.
Volume confirmation requires institutional participation, filtering out low-conviction moves.
Target trade frequency: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R(14) calculation
    window = 14
    highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    williams_r = -100 * (highest_high - close) / denominator
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, window, 20)  # EMA50 needs 50, Williams %R needs 14, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema50_val = ema50_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long: Oversold (WR < -80) AND uptrend (price > EMA50) AND volume spike (1.8x avg)
            if wr < -80 and price > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (WR > -20) AND downtrend (price < EMA50) AND volume spike (1.8x avg)
            elif wr > -20 and price < ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses above -50 (for long) or below -50 (for short)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_14_1dEMA50_Trend_VolumeConfirmation_ExitAt_Minus50"
timeframe = "6h"
leverage = 1.0